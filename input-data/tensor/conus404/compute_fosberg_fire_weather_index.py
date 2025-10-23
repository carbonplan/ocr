# COILED n-tasks 1
# COILED --region us-west-2
# COILED --tag Project=OCR

import time
from typing import Any

import coiled
import dask.base
import icechunk
import icechunk.xarray
import rich
import typer
import upath
import xarray as xr
from odc.geo import CRS
from odc.geo.xr import assign_crs, xr_reproject

from ocr import catalog
from ocr.conus404 import (
    compute_relative_humidity,
    compute_wind_speed_and_direction,
    load_conus404,
    rotate_winds_to_earth,
)
from ocr.risks.fire import fosberg_fire_weather_index
from ocr.utils import geo_sel

console = rich.console.Console()
app = typer.Typer(help='Compute Fosberg Fire Weather Index from CONUS404 data.')


def reproject(
    src_dataset: xr.Dataset,
    src_crs_wkt: str,
    *,
    chunk_lat: int = 6000,
    chunk_lon: int = 4500,
) -> xr.Dataset:
    """Reproject the wind direction distributions to the geobox of a target dataset."""
    target_dataset_name = 'USFS-wildfire-risk-communities-4326'
    tgt = catalog.get_dataset(target_dataset_name).to_xarray().astype('float32')
    tgt = assign_crs(tgt, crs='EPSG:4326')
    geobox = tgt.odc.geobox

    src_crs = CRS(src_crs_wkt)
    src_dataset = assign_crs(src_dataset, crs=src_crs)
    result = (
        xr_reproject(
            src_dataset,
            geobox,
            resampling='nearest',
        )
        .astype('float32')
        .chunk({'latitude': chunk_lat, 'longitude': chunk_lon})
    )

    # To avoid issues with floating point noise in coordinates, we directly adopt the target dataset's coords
    # fixes https://github.com/carbonplan/ocr/issues/247
    result = result.assign_coords(latitude=tgt.latitude, longitude=tgt.longitude)

    # sort the coordinates to ensure ascending order
    result = result.sortby(['latitude', 'longitude'])

    result.attrs.update({'reprojected_to': target_dataset_name})
    return result


def setup_cluster(
    name: str,
    region: str = 'us-west-2',
    software: str | None = None,
    min_workers: int = 2,
    max_workers: int = 50,
    worker_vm_types: str = 'm8g.2xlarge',
    scheduler_vm_types: str = 'm8g.large',
) -> coiled.Cluster | None:
    """Create and return a Coiled cluster (or None if min_workers == 0).

    Returns the cluster object; a dask `Client` can be obtained via
    `cluster.get_client()`.
    """
    if min_workers <= 0:
        console.log('Skipping cluster creation (min_workers <= 0)')
        return None

    args = {
        'name': name,
        'region': region,
        'n_workers': [min_workers, max_workers],
        'tags': {'Project': 'OCR'},
        'worker_vm_types': worker_vm_types,
        'scheduler_vm_types': scheduler_vm_types,
        'spot_policy': 'spot_with_fallback',
        'software': software,
    }
    console.log(f'Creating Coiled cluster with args: {args}')
    cluster = coiled.Cluster(**args)
    # Touch the client to ensure startup
    client = cluster.get_client()
    console.log(
        f'Cluster {cluster.name} started with {len(client.scheduler_info()["workers"])} workers (initial).'
    )
    return cluster


def _write_icechunk_store(
    *,
    out_path: str,
    dataset: xr.Dataset,
    commit_message: str,
    overwrite: bool,
):
    """Write dataset to an Icechunk repo and perform housekeeping.

    Steps:
    - Open or create repository at out_path
    - Skip if commit already present and not overwriting
    - Write data via a writable session
    - Rebase with BasicConflictSolver and commit
    - Expire old snapshots and garbage collect pieces
    """
    path = upath.UPath(out_path)
    protocol = path.protocol
    if protocol == 's3':
        parts = path.parts
        bucket = parts[0].strip('/')
        prefix = '/'.join(parts[1:])
        storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
    elif protocol in {'file', 'local'} or protocol == '':
        storage = icechunk.local_filesystem_storage(path=str(path))
    else:
        raise ValueError(f'Unsupported protocol: {protocol}')
    repo = icechunk.Repository.open_or_create(storage)

    # Inspect history
    messages = [commit.message for commit in repo.ancestry(branch='main')]
    if commit_message in messages and not overwrite:
        console.log(f'Data already committed to {out_path}, skipping save (overwrite=False).')
        return

    if overwrite and commit_message in messages:
        console.log(f'Overwriting existing data at {out_path}...')
        # If a full reset is desired, uncomment the following line and pick an init commit
        # init_commit = list(repo.ancestry(branch="main"))[-1]
        # repo.reset_branch('main', init_commit.id)

    # Write data
    session = repo.writable_session('main')
    console.log(f'Saving dataset to {out_path}...')
    icechunk.xarray.to_icechunk(
        dataset,
        session,
        mode='w',
    )

    # Rebase and commit
    session.rebase(
        icechunk.BasicConflictSolver(on_chunk_conflict=icechunk.VersionSelection.UseOurs)
    )
    session.commit(commit_message)

    # Housekeeping: expire old snapshots and GC pieces
    latest_commit = list(repo.ancestry(branch='main'))[0]
    console.log(latest_commit)
    expired = repo.expire_snapshots(older_than=latest_commit.written_at)
    console.log(f'{out_path} Expired {len(expired)} old snapshots.')
    results = repo.garbage_collect(latest_commit.written_at)
    console.log(f'{out_path} Garbage collection results: {results}')


def _open_icechunk_dataset(path: str) -> xr.Dataset:
    """Open an Icechunk repo path (s3 or local) as an xarray.Dataset (readonly).

    Parameters
    ----------
    path : str
        Path to the Icechunk repository. Supports s3:// and local paths.

    Returns
    -------
    xr.Dataset
        Opened dataset using zarr engine with no eager loading (lazy dask arrays).
    """
    up = upath.UPath(path)
    protocol = up.protocol
    if protocol == 's3':
        parts = up.parts
        bucket = parts[0].strip('/')
        prefix = '/'.join(parts[1:])
        storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
    elif protocol in {'file', 'local'} or protocol == '':
        storage = icechunk.local_filesystem_storage(path=str(up))
    else:
        raise ValueError(f'Unsupported protocol: {protocol}')

    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session('main')
    store: Any = session.store
    ds = xr.open_dataset(store, engine='zarr', consolidated=False, chunks={})
    return ds


@app.command('compute')
def compute_ffwi(
    dry_run: bool = typer.Option(
        False, help='If true, do not create cluster or run full computations. Run a test instead.'
    ),
    overwrite: bool = typer.Option(False, help='If true, overwrite existing output.'),
    output_base: str = typer.Option(
        's3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi', help='Base output path.'
    ),
    min_workers: int = typer.Option(10, help='Minimum number of Coiled workers'),
    max_workers: int = typer.Option(70, help='Maximum number of Coiled workers'),
    worker_vm_types: str = typer.Option('m8g.2xlarge', help='Worker VM type'),
    software: str | None = typer.Option(None, help='Coiled software environment ID'),
):
    """Compute and save the Fosberg Fire Weather Index (FFWI) from CONUS404 data.

    This command computes only the base FFWI dataset and writes it to Icechunk.
    Use the `postprocess` command to compute quantiles and modes later without
    recomputing FFWI.
    """
    start = time.time()
    cluster = None
    if not dry_run:
        cluster = setup_cluster(
            name='ffwi-computation',
            min_workers=min_workers,
            max_workers=max_workers,
            worker_vm_types=worker_vm_types,
            scheduler_vm_types='m8g.large',
            software=software,
        )
        if cluster is None:
            raise RuntimeError('Cluster setup failed or was skipped (min_workers <= 0).')
        client = cluster.get_client()
        console.log(f'Dask client: {client}')
        console.log('Waiting 60 seconds for cluster to stabilize...')

    ds = load_conus404(add_spatial_constants=True)
    if dry_run:
        console.log('Running in dry run mode. Skipping cluster creation and full computations.')
        ds = geo_sel(ds, bbox=(-120.1, 39.1, -120, 39))

    console.log(f'Loaded CONUS404 data: {ds}')

    # compute relative humidity
    console.log('Computing relative humidity from specific humidity and temperature...')
    hurs = compute_relative_humidity(ds)
    earth_u, earth_v = rotate_winds_to_earth(ds)
    wind_ds = compute_wind_speed_and_direction(earth_u, earth_v)

    ffwi = fosberg_fire_weather_index(
        hurs=hurs, T2=ds['T2'], sfcWind=wind_ds['sfcWind']
    ).to_dataset()
    ffwi = ffwi.chunk({'x': 10, 'y': 10})
    console.log(f'Computed Fosberg Fire Weather Index: {ffwi}')
    console.log(ffwi)
    # save to icechunk zarr (full FFWI)
    out_path = f'{output_base}/fosberg-fire-weather-index.icechunk'
    _write_icechunk_store(
        out_path=out_path,
        dataset=ffwi,
        commit_message='Add Fosberg Fire Weather Index data.',
        overwrite=overwrite,
    )

    # save winds
    wind_out_path = f'{output_base}/winds.icechunk'
    wind_ds = wind_ds.chunk({'x': 10, 'y': 10})
    _write_icechunk_store(
        out_path=wind_out_path,
        dataset=wind_ds,
        commit_message='Add surface wind data used in Fosberg Fire Weather Index computation.',
        overwrite=overwrite,
    )

    if cluster is not None:
        console.log('Closing Coiled cluster')
        cluster.close()

    console.log(f'Completed in {(time.time() - start) / 60:.2f} minutes')
    console.rule('Done')


@app.command('postprocess')
def postprocess_ffwi(
    dry_run: bool = typer.Option(
        False, help='If true, do not create cluster or run full computations. Run a test instead.'
    ),
    overwrite: bool = typer.Option(False, help='If true, overwrite existing output.'),
    output_base: str = typer.Option(
        's3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi', help='Base output path.'
    ),
    quantiles: list[float] = typer.Option([0.99], help='Quantiles to compute and save.'),
    min_workers: int = typer.Option(10, help='Minimum number of Coiled workers'),
    max_workers: int = typer.Option(70, help='Maximum number of Coiled workers'),
    worker_vm_types: str = typer.Option('m8g.2xlarge', help='Worker VM type'),
    software: str | None = typer.Option(None, help='Coiled software environment ID'),
):
    """Compute quantiles and wind-direction mode using a precomputed FFWI store.

    This command loads an existing FFWI dataset (produced by `compute`) and computes
    quantiles and, optionally, the prevailing wind direction (mode) for high-FFWI periods.
    """
    start = time.time()
    cluster = None
    if not dry_run:
        cluster = setup_cluster(
            name='ffwi-postprocess',
            min_workers=min_workers,
            max_workers=max_workers,
            worker_vm_types=worker_vm_types,
            scheduler_vm_types='m8g.large',
            software=software,
        )
        if cluster is None:
            raise RuntimeError('Cluster setup failed or was skipped (min_workers <= 0).')
        client = cluster.get_client()
        console.log(f'Dask client: {client}')
        console.log('Waiting 60 seconds for cluster to stabilize...')

    # Load stored FFWI
    ffwi_path = f'{output_base}/fosberg-fire-weather-index.icechunk'
    console.log(f'Loading FFWI from {ffwi_path}...')
    ffwi = _open_icechunk_dataset(ffwi_path)

    # Quantiles and optional mode
    console.log(f'Postprocessing FFWI: {ffwi}')
    for quantile in quantiles:
        q_out_path = f'{output_base}/fosberg-fire-weather-index-p{int(quantile * 100)}.icechunk'
        console.log(f'Computing and saving {quantile} quantile to {q_out_path}...')
        ffwi_quantile = ffwi.quantile(quantile, dim='time').chunk({'x': -1, 'y': -1})
        ffwi_quantile = dask.base.optimize(ffwi_quantile)[0]
        # ffwi_quantile = ffwi_quantile.persist()
        ffwi_quantile.FFWI.attrs['description'] = (
            f'Fosberg Fire Weather Index {quantile} quantile over time dimension.'
        )
        ffwi_quantile.FFWI.attrs['computed_for_quantile'] = quantile
        _write_icechunk_store(
            out_path=q_out_path,
            dataset=ffwi_quantile,
            commit_message=f'Add Fosberg Fire Weather Index {quantile} quantile data.',
            overwrite=overwrite,
        )
        console.log(f'Saved {quantile} quantile to {q_out_path}.')

    if cluster is not None:
        console.log('Closing Coiled cluster')
        cluster.close()

    console.log(f'Completed in {(time.time() - start) / 60:.2f} minutes')
    console.rule('Done')


@app.command('reproject')
def reproject_ffwi(
    distribution_input_path: str = typer.Option(
        's3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index-p99-wind-direction-distribution.icechunk',
        help='Input path to the wind direction distribution Icechunk repository.',
    ),
    distribution_output_path: str = typer.Option(
        's3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index-p99-wind-direction-distribution-30m-4326.icechunk',
        help='Output path for the reprojected wind direction distribution Icechunk repository.',
    ),
    overwrite: bool = typer.Option(False, help='If true, overwrite existing output.'),
    dry_run: bool = typer.Option(
        False, help='If true, do not create cluster or run full computations. Run a test instead.'
    ),
    min_workers: int = typer.Option(10, help='Minimum number of Coiled workers'),
    max_workers: int = typer.Option(70, help='Maximum number of Coiled workers'),
    worker_vm_types: str = typer.Option('m8g.2xlarge', help='Worker VM type'),
    software: str | None = typer.Option(None, help='Coiled software environment ID'),
):
    """Reproject the FFWI mode and corresponding wind direction distribution datasets.

    Both the modal wind direction (already summarized) and the full wind direction
    distribution (used to derive the mode) are reprojected onto the geobox of the
    target dataset, currently hard-coded to the catalog entry
    'USFS-wildfire-risk-communities-4326'.
    """
    start = time.time()
    cluster = None
    if not dry_run:
        cluster = setup_cluster(
            name='ffwi-postprocess',
            min_workers=min_workers,
            max_workers=max_workers,
            worker_vm_types=worker_vm_types,
            scheduler_vm_types='m8g.large',
            software=software,
        )
        if cluster is None:
            raise RuntimeError('Cluster setup failed or was skipped (min_workers <= 0).')
        client = cluster.get_client()
        console.log(f'Dask client: {client}')
        console.log('Waiting 60 seconds for cluster to stabilize...')

    console.log(f'Loading wind direction distribution from {distribution_input_path}...')
    wind_dir_distribution = _open_icechunk_dataset(distribution_input_path)
    console.log(f'Loaded wind direction distribution: {wind_dir_distribution}')

    conus404 = load_conus404(add_spatial_constants=True)
    src_crs_wkt = conus404['crs'].attrs['crs_wkt']
    console.log(f'CONUS404 CRS WKT: {src_crs_wkt}')

    reprojected_distribution = reproject(
        wind_dir_distribution,
        src_crs_wkt,
    )
    console.log(f'Reprojected wind direction distribution: {reprojected_distribution}')

    _write_icechunk_store(
        out_path=distribution_output_path,
        dataset=reprojected_distribution,
        commit_message='Reprojected wind direction distribution to 30m EPSG:4326 geobox',
        overwrite=overwrite,
    )

    if cluster is not None:
        console.log('Closing Coiled cluster')
        cluster.close()

    console.log(f'Completed in {(time.time() - start) / 60:.2f} minutes')
    console.rule('Done')


if __name__ == '__main__':
    app()
