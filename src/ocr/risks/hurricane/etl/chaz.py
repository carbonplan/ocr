"""Input data ETL for CHAZ tropical-cyclone wind hazard (Meiler et al. 2026, doi:10.5061/dryad.qfttdz0vz).

Ported from carbonplan/ocr branch Shane98c/chaz (chaz_matrix.py). Reads the raw netCDFs under
fetch.RAW_PREFIX and writes one group per (metric, representation) to STORE, dims
(scenario, period, variant, gcm, lat, lon), one float32 variable per band, EPSG:4326.
Missing files stay NaN at their fixed index, so a GCM's position never shifts.

Representations: 'points' snaps the native coastal point files (300 arcsec land, 3600 arcsec
ocean) to the nearest GRIDS node; 'raster' reads the published 0.05 deg interpolation as-is.

Writes an empty template, then fills it one shard block at a time (region='auto'), in one commit.
An existing group raises unless --overwrite, which replaces that group only. All runs commit to
one repo: run them one after another.
    pixi run hurricane-etl --metric return_periods [--representation raster] [--prefix scratch/<you>] [--overwrite]
ERA5-forced baseline: --source era5 writes <group>_era5, dims (lat, lon) on its own grid.
On Coiled: coiled batch run --vm-type r8g.2xlarge --arm --tag Project=OCR -- python -m ocr.risks.hurricane.etl.chaz --metric exceedance_intensity
"""

from collections.abc import Iterator
from itertools import product
from typing import Annotated, Literal

import dask.array as da
import icechunk
import numpy as np
import obstore
import typer
import xarray as xr
import xproj  # noqa: F401
import zarr
from obspec_utils.readers import EagerStoreReader
from topozarr import attach_geozarr_metadata

from ocr.config import INPUT_PREFIX, StorageConfig, _check_prefix
from ocr.risks.hurricane.etl.fetch import RAW_PREFIX, Metric

STORE = 'hurricane/chaz.icechunk'  # under --prefix (default input)
GCMS = ['CESM2', 'CNRM-CM6-1', 'EC-Earth3', 'IPSL-CM6A-LR', 'MIROC6', 'UKESM1-0-LL']
SCENARIOS = ['ssp245', 'ssp370', 'ssp585']
PERIODS = ['base', 'fut1', 'fut2']
VARIANTS = ['CRH', 'SD']
BANDS: dict[Metric, list[str]] = {
    'return_periods': ['thr_33', 'thr_50'],
    'exceedance_intensity': ['rp_10', 'rp_25', 'rp_50', 'rp_100', 'rp_250', 'rp_1000'],
}
COORDS = {'scenario': SCENARIOS, 'period': PERIODS, 'variant': VARIANTS, 'gcm': GCMS}
DIMS = (*COORDS, 'lat', 'lon')

Rep = Literal['points', 'raster']
Source = Literal['gcm', 'era5']


def _axis(start: float, n: int, step: float) -> np.ndarray:
    return start + np.arange(n) * step


# (rep, source) -> (lat, lon) of the published CHAZ data. Point grids span the data at 300 arcsec;
# ERA5 points sit half a cell off the GCM ones. Raster grids drop the repeated 180 meridian.
GRIDS: dict[tuple[Rep, Source], tuple[np.ndarray, np.ndarray]] = {
    ('points', 'gcm'): (_axis(-59, 1429, 1 / 12), _axis(-179.5, 4309, 1 / 12)),
    ('points', 'era5'): (_axis(-59.5 - 1 / 24, 1435, 1 / 12), _axis(-180 - 1 / 24, 4321, 1 / 12)),
    ('raster', 'gcm'): (_axis(-59, 2381, 0.05), _axis(-180, 7200, 0.05)),
    ('raster', 'era5'): (_axis(-60, 2401, 0.05), _axis(-180, 7200, 0.05)),
}

# spacetimer.recommend(template, 'space') @ 3c48ac9 on GRIDS, scenario forced to 1 in chunks/shards.
# Dims as in DIMS; ERA5 groups are (lat, lon).
ENCODING: dict[tuple[Rep, Source], dict[str, tuple[int, ...]]] = {
    ('points', 'gcm'): {'chunks': (1, 1, 1, 1, 1429, 2155), 'shards': (1, 3, 2, 6, 1429, 4310)},
    ('points', 'era5'): {'chunks': (1435, 2161), 'shards': (1435, 4322)},
    ('raster', 'gcm'): {'chunks': (1, 1, 1, 1, 1191, 1800), 'shards': (1, 1, 2, 6, 2382, 7200)},
    ('raster', 'era5'): {'chunks': (1201, 1800), 'shards': (2402, 7200)},
}


def _layout(metric: str, rep: Rep) -> tuple[str, str]:
    """(raw dir under RAW_PREFIX, filename suffix) of one metric/representation."""
    if rep == 'points':
        return f'{RAW_PREFIX}/{metric}/nc', metric
    return f'{RAW_PREFIX}/{metric}/raster.nc', f'{metric}_raster'


def gcm_key(metric: str, gcm: str, scenario: str, period: str, variant: str, rep: Rep) -> str:
    root, suffix = _layout(metric, rep)
    return (
        f'{root}/per-GCM/{gcm}/{scenario}/'
        f'TC_global_0300as_CHAZ_{gcm}_{period}_{scenario}_80ens_{variant}_H08_{suffix}.nc'
    )


def era5_key(metric: str, rep: Rep) -> str:
    root, suffix = _layout(metric, rep)
    return f'{root}/ERA5/TC_global_0300as_CHAZ_ERA5_{suffix}.nc'


def group_name(metric: str, rep: Rep, era5: bool = False) -> str:
    return (metric if rep == 'points' else f'{metric}_raster') + ('_era5' if era5 else '')


def band_attrs(metric: str, band: str) -> dict[str, str]:
    """long_name/units as in the raw files (lost when the cube is assembled, so set here)."""
    n = band.split('_')[1]
    if metric == 'return_periods':
        return {'long_name': f'Return period for intensity ≥ {n} m/s', 'units': 'years'}
    return {'long_name': f'Exceedance intensity for RP={n} years', 'units': 'm/s'}


def _snap(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Nearest axis node of each value; KeyError if one falls off the axis. ERA5's 1 deg ocean points
    sit exactly between nodes: np.round rounds half to even, which reproduces the v1 store exactly."""
    i = np.round((values - axis[0]) / (axis[1] - axis[0])).astype(int)
    if (off := (i < 0) | (i >= axis.size)).any():
        raise KeyError(f'{int(off.sum())} values off the axis [{axis[0]}, {axis[-1]}]')
    return axis[i]


def load(cfg: StorageConfig, key: str, metric: Metric, rep: Rep, source: Source) -> xr.Dataset:
    """One raw file as float32 bands on GRIDS[rep, source], NaN where a point file has no point."""
    lat, lon = GRIDS[rep, source]
    bands = BANDS[metric]
    with EagerStoreReader(cfg.object_store(), key) as f:
        ds = xr.open_dataset(f, engine='h5netcdf').load()  # load before the reader closes
    if rep == 'raster' and source == 'era5':
        ds = ds.rename({f'{b.split("_")[0]}_{b}': b for b in bands})  # thr_thr_33, rp_rp_10
    ds = ds[bands]
    if rep == 'raster':
        # Exact coords: drops float noise and the repeated 180 meridian, KeyError on another grid.
        ds = ds.sel(lat=lat, lon=lon, method='nearest', tolerance=1e-6)
        return ds.assign_coords(lat=lat, lon=lon).astype('float32')
    snapped = {
        'lat': ('points', _snap(ds.lat.values, lat)),
        'lon': ('points', _snap(ds.lon.values, lon)),
    }
    ds = ds.assign_coords(snapped).set_index(points=['lat', 'lon'])
    # Real files have a few points sharing a cell (69 in ERA5 points); last wins, as in the v1 store.
    if n := int(ds.indexes['points'].duplicated().sum()):
        print(f'    note: {key}: {n} points share a cell; last value kept')
        ds = ds.drop_duplicates('points', keep='last')
    return ds.unstack('points').reindex(lat=lat, lon=lon).astype('float32')


def template(metric: Metric, coords: dict[str, list[str] | np.ndarray]) -> xr.Dataset:
    """All-NaN float32 bands over coords, one numpy buffer per band."""
    shape = tuple(len(v) for v in coords.values())
    data = {b: (tuple(coords), np.full(shape, np.nan, dtype='float32')) for b in BANDS[metric]}
    return xr.Dataset(data, coords=coords)


def _blocks(shards: tuple[int, ...]) -> Iterator[dict[str, list[str]]]:
    """COORDS of each shard, so each shard is written once. Assumes lat/lon shards span the grid."""
    axes = {
        d: [vals[i : i + n] for i in range(0, len(vals), n)]
        for (d, vals), n in zip(COORDS.items(), shards[: len(COORDS)], strict=True)
    }
    for combo in product(*axes.values()):
        yield dict(zip(axes, combo))


def _finalize(ds: xr.Dataset, metric: Metric, rep: Rep, source: Source) -> xr.Dataset:
    """Band and dataset attrs, CRS and GeoZarr metadata."""
    ds = ds.copy()  # shallow: attrs below must not leak into the caller's dataset
    for b in BANDS[metric]:
        ds[b].attrs = band_attrs(metric, b)
    ds.attrs = {
        'title': f'CHAZ tropical-cyclone {metric} ({rep})',
        'source': 'https://datadryad.org/dataset/doi:10.5061/dryad.qfttdz0vz',
        'source_doi': 'https://doi.org/10.5061/dryad.qfttdz0vz',
        'paper_doi': 'https://doi.org/10.1038/s41597-025-06452-0',
        'license': 'CC0-1.0',
        'representation': rep,
        'bands': BANDS[metric],  # zarr does not keep array order; this is the canonical order
        **({'forcing': 'ERA5'} if source == 'era5' else {}),
    }
    ds = ds.proj.assign_crs(spatial_ref='EPSG:4326').proj.write_crs_info()
    return attach_geozarr_metadata(ds, x_dim='lon', y_dim='lat')


def run(
    metric: Annotated[Metric, typer.Option()],
    rep: Annotated[Rep, typer.Option('--representation')] = 'points',
    source: Source = 'gcm',
    prefix: str = INPUT_PREFIX,
    overwrite: bool = False,
) -> None:
    cfg = StorageConfig()
    store = f'{_check_prefix(prefix)}/{STORE}'  # raw inputs always read from RAW_PREFIX
    group = group_name(metric, rep, era5=source == 'era5')
    enc = ENCODING[rep, source]
    repo = icechunk.Repository.open_or_create(cfg.icechunk_storage(store))
    session = repo.writable_session('main')
    if group in zarr.open_group(session.store, mode='a') and not overwrite:
        raise FileExistsError(f'{cfg.uri(store)} group {group} exists; pass --overwrite')
    write = {'group': group, 'consolidated': False, 'zarr_format': 3}

    if source == 'era5':
        ds = _finalize(load(cfg, era5_key(metric, rep), metric, rep, source), metric, rep, source)
        ds.to_zarr(session.store, mode='w', encoding=dict.fromkeys(BANDS[metric], enc), **write)
        print(f'wrote {cfg.uri(store)} group {group} -> {session.commit(group)}')
        return

    raw_dir = f'{_layout(metric, rep)[0]}/per-GCM/'
    present = {m['path'] for batch in obstore.list(cfg.object_store(), raw_dir) for m in batch}
    if not present:
        raise FileNotFoundError(f'no raw {group} files under {cfg.uri(raw_dir)}')
    lat, lon = GRIDS[rep, source]
    coords = COORDS | {'lat': lat, 'lon': lon}
    shape = tuple(len(v) for v in coords.values())
    # Lazy, never computed (compute=False): only writes metadata; missing chunks read as NaN.
    lazy = da.full(shape, np.nan, dtype='float32', chunks=enc['shards'])
    tmpl = xr.Dataset({b: (DIMS, lazy) for b in BANDS[metric]}, coords=coords)
    _finalize(tmpl, metric, rep, source).to_zarr(
        session.store, mode='w', compute=False, encoding=dict.fromkeys(BANDS[metric], enc), **write
    )

    for block in _blocks(enc['shards']):
        ds = template(metric, block | {'lat': lat, 'lon': lon})
        found = 0
        for combo in product(*block.values()):
            sel = dict(zip(block, combo, strict=True))
            if (key := gcm_key(metric, rep=rep, **sel)) in present:
                ds.loc[sel] = load(cfg, key, metric, rep, source)
                found += 1
        if found:
            ds.to_zarr(session.store, region='auto', **write)
        print(f'  {found} files -> {block}')
    print(f'wrote {cfg.uri(store)} group {group} -> {session.commit(group)}')


if __name__ == '__main__':
    typer.run(run)
