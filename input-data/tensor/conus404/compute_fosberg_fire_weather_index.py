import coiled
import icechunk
import icechunk.xarray
import rich
import typer

from ocr.conus404 import (
    compute_relative_humidity,
    compute_wind_speed_and_direction,
    geo_sel,
    load_conus404,
    rotate_winds_to_earth,
)
from ocr.risks.fire import fosberg_fire_weather_index

console = rich.console.Console()
app = typer.Typer(help='Compute Fosberg Fire Weather Index from CONUS404 data.')


def setup_cluster(
    name: str,
    region: str = 'us-west-2',
    min_workers: int = 2,
    max_workers: int = 50,
    worker_vm_types: str = 'm8g.xlarge',
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
    }
    console.log(f'Creating Coiled cluster with args: {args}')
    cluster = coiled.Cluster(**args)
    # Touch the client to ensure startup
    client = cluster.get_client()
    console.log(
        f'Cluster {cluster.name} started with {len(client.scheduler_info()["workers"])} workers (initial).'
    )
    return cluster


@app.command()
def main(
    dry_run: bool = typer.Option(
        False, help='If true, do not create cluster or run full computations. Run a test instead.'
    ),
    overwrite: bool = typer.Option(False, help='If true, overwrite existing output.'),
    output_base: str = typer.Option('/tmp', help='Base output path.'),
):
    """Compute Fosberg Fire Weather Index from CONUS404 data."""
    ds = load_conus404(add_spatial_constants=True)
    if dry_run:
        console.log('Running in dry run mode. Skipping cluster creation and full computations.')
        ds = geo_sel(ds, bbox=(-120.1, 39.1, -120, 39)).compute()
    console.log(f'Loaded CONUS404 data: {ds}')

    # compute relative humidity
    console.log('Computing relative humidity from specific humidity and temperature...')
    hurs = compute_relative_humidity(ds)
    earth_u, earth_v = rotate_winds_to_earth(ds)
    wind_ds = compute_wind_speed_and_direction(earth_u, earth_v)

    ffwi = fosberg_fire_weather_index(
        hurs=hurs, T2=ds['T2'], sfcWind=wind_ds['sfcWind']
    ).to_dataset()
    console.log(f'Computed Fosberg Fire Weather Index: {ffwi}')
    console.log(ffwi)
    # save to icechunk zarr
    out_path = f'{output_base}/ffwi.icechunk'
    storage = icechunk.local_filesystem_storage(out_path)
    repo = icechunk.Repository.open_or_create(storage)
    messages = []
    for commit in repo.ancestry(branch='main'):
        messages.append(commit.message)

    console.print(messages)
    if 'Add Fosberg Fire Weather Index data.' in messages and not overwrite:
        console.log(
            f'Fosberg Fire Weather Index data already committed to {out_path}, skipping save.'
        )
        return

    else:
        # wipe out existing data if overwrite is True
        if overwrite:
            console.log(f'Overwriting existing data at {out_path}...')
            #  repo.reset_branch('main', init_commit.id)

    session = repo.writable_session('main')
    console.log(f'Saving Fosberg Fire Weather Index to {out_path}...')
    icechunk.xarray.to_icechunk(
        ffwi,
        session,
        mode='w',
    )
    session.rebase(
        icechunk.BasicConflictSolver(on_chunk_conflict=icechunk.VersionSelection.UseOurs)
    )

    session.commit(
        'Add Fosberg Fire Weather Index data.',
    )

    latest_commit = list(repo.ancestry(branch='main'))[0]
    console.log(latest_commit)
    expired = repo.expire_snapshots(older_than=latest_commit.written_at)
    console.log(f'Expired {len(expired)} old snapshots.')
    # delete data associated with expired snapshots
    results = repo.garbage_collect(latest_commit.written_at)
    console.log(f'{results}')


if __name__ == '__main__':
    app()
