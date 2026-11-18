"""Input data ETL for CHAZ tropical-cyclone wind hazard (Meiler et al. 2026, doi:10.5061/dryad.qfttdz0vz).

Ported from carbonplan/ocr branch Shane98c/chaz (chaz_matrix.py). Reads the raw netCDFs under
fetch.RAW_PREFIX and writes one group per (metric, representation) to STORE, dims
(scenario, period, variant, gcm, lat, lon), one float32 variable per band, EPSG:4326.
Missing files stay NaN at their fixed index, so a GCM's position never shifts.

Representations: 'points' scatters the native coastal point files (300 arcsec land, 3600 arcsec
ocean) onto the lattice inferred from their coordinates; 'raster' reads the published 0.05 deg
interpolation as-is.

Writes one commit per scenario (one shard block). A rerun resumes from the last committed scenario;
--overwrite rebuilds that group only. All runs commit to one repo: run them one after another.
    pixi run hurricane-etl --metric return_periods [--representation raster] [--prefix scratch/<you>] [--overwrite]
ERA5-forced baseline: --source era5 writes <group>_era5, dims (lat, lon) on its own grid, one commit.
On Coiled: coiled batch run --vm-type r8g.2xlarge --arm --tag Project=OCR -- python -m ocr.risks.hurricane.etl.chaz --metric exceedance_intensity
"""

import warnings
from typing import Annotated, Literal

import dask.array as da
import icechunk
import numpy as np
import obstore
import spacetimer
import typer
import xarray as xr
import xproj  # noqa: F401
from obspec_utils.readers import BufferedStoreReader, EagerStoreReader
from topozarr import attach_geozarr_metadata

from ocr.config import StorageConfig, _check_prefix
from ocr.risks.hurricane.etl.fetch import RAW_PREFIX

STORE = 'hurricane/chaz.icechunk'  # under --prefix (default input)
GCMS = ['CESM2', 'CNRM-CM6-1', 'EC-Earth3', 'IPSL-CM6A-LR', 'MIROC6', 'UKESM1-0-LL']
SCENARIOS = ['ssp245', 'ssp370', 'ssp585']
PERIODS = ['base', 'fut1', 'fut2']
VARIANTS = ['CRH', 'SD']
BANDS = {
    'return_periods': ['thr_33', 'thr_50'],
    'exceedance_intensity': ['rp_10', 'rp_25', 'rp_50', 'rp_100', 'rp_250', 'rp_1000'],
}
DIMS = ('scenario', 'period', 'variant', 'gcm', 'lat', 'lon')
RES = 1.0 / 12.0  # native CHAZ GCM grid, 300 arcsec; fallback step

Metric = Literal['return_periods', 'exceedance_intensity']
Rep = Literal['points', 'raster']


def gcm_key(metric: str, gcm: str, scenario: str, period: str, variant: str, rep: Rep) -> str:
    subdir, suffix = ('nc', metric) if rep == 'points' else ('raster.nc', f'{metric}_raster')
    return (
        f'{RAW_PREFIX}/{metric}/{subdir}/per-GCM/{gcm}/{scenario}/'
        f'TC_global_0300as_CHAZ_{gcm}_{period}_{scenario}_80ens_{variant}_H08_{suffix}.nc'
    )


def era5_key(metric: str, rep: Rep) -> str:
    subdir, suffix = ('nc', metric) if rep == 'points' else ('raster.nc', f'{metric}_raster')
    return f'{RAW_PREFIX}/{metric}/{subdir}/ERA5/TC_global_0300as_CHAZ_ERA5_{suffix}.nc'


def group_name(metric: str, rep: Rep, era5: bool = False) -> str:
    return (metric if rep == 'points' else f'{metric}_raster') + ('_era5' if era5 else '')


def _detect_axis(coords: np.ndarray) -> tuple[float, float]:
    """(step, phase) of a coordinate axis from its point values: the modal spacing, anchored at the
    phase shared by most points. Anchoring at the raw minimum instead puts most ERA5 points between
    two rows, collides ~41% of them and leaves NaN bands. Off-phase minority snaps to nearest node."""
    a = np.unique(np.round(np.asarray(coords, dtype='float64').ravel(), 6))
    d = np.diff(a)
    d = d[d > 1e-6]
    if d.size == 0:
        return RES, 0.0
    vals, counts = np.unique(np.round(d, 6), return_counts=True)
    step = float(vals[int(np.argmax(counts))])
    inv = round(1.0 / step)
    if inv and abs(1.0 / inv - step) < 1e-4:
        step = 1.0 / inv  # snap to an exact 1/n so phases are clean
    frac = np.round(np.mod(np.asarray(coords, dtype='float64').ravel() / step, 1.0), 3)
    frac[frac > 0.999] = 0.0  # values just under a node wrap to 0
    fv, fc = np.unique(frac, return_counts=True)
    phase = float(fv[int(np.argmax(fc))])
    return step, phase


def _axis_lattice(coords: np.ndarray, step: float, phase: float) -> np.ndarray:
    """Regular lattice with nodes at (k + phase) * step spanning the data."""
    k0 = int(np.floor(float(np.min(coords)) / step - phase))
    k1 = int(np.ceil(float(np.max(coords)) / step - phase))
    return (np.arange(k0, k1 + 1) + phase) * step


def _scatter(ds: xr.Dataset, band: str, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Point file -> (lat, lon) grid, NaN where no point, snapping each point to the lattice."""
    lon, lat = ds['lon'].values, ds['lat'].values
    ix = np.clip(np.round((lon - lons[0]) / (lons[1] - lons[0])).astype(int), 0, lons.size - 1)
    iy = np.clip(np.round((lat - lats[0]) / (lats[1] - lats[0])).astype(int), 0, lats.size - 1)
    collisions = lon.size - np.unique(iy.astype(np.int64) * lons.size + ix).size
    if collisions:
        print(f'    note: {collisions}/{lon.size} {band} off-phase points snapped')
    grid = np.full((lats.size, lons.size), np.nan, dtype='float32')
    grid[iy, ix] = ds[band].values.astype('float32')
    return grid


def present_keys(cfg: StorageConfig, metric: str, rep: Rep) -> set[str]:
    subdir = 'nc' if rep == 'points' else 'raster.nc'
    store = cfg.object_store()
    return {
        m['path']
        for batch in obstore.list(
            store, f'{RAW_PREFIX}/{metric}/{subdir}/per-GCM/'
        )  # not ERA5/: own grid
        for m in batch
    }


def read_nc(cfg: StorageConfig, key: str) -> xr.Dataset:
    with EagerStoreReader(cfg.object_store(), key) as f:
        return xr.open_dataset(f, engine='h5netcdf').load()


def grid(cfg: StorageConfig, keys: list[str], rep: Rep) -> tuple[np.ndarray, np.ndarray]:
    """(lats, lons) shared by every file of one metric/representation."""
    store = cfg.object_store()
    if rep == 'raster':
        with BufferedStoreReader(store, keys[0]) as f, xr.open_dataset(f, engine='h5netcdf') as d:
            return _raster_coords(d)
    lat_all, lon_all = [], []
    for key in keys:  # range reads: only lat/lon leave S3
        with BufferedStoreReader(store, key, buffer_size=8 << 20) as f:
            with xr.open_dataset(f, engine='h5netcdf') as d:
                lat_all.append(d['lat'].values)
                lon_all.append(d['lon'].values)
    lats, lons = np.concatenate(lat_all), np.concatenate(lon_all)
    return _axis_lattice(lats, *_detect_axis(lats)), _axis_lattice(lons, *_detect_axis(lons))


def _regular(a: np.ndarray) -> np.ndarray:
    """Axis rebuilt as start + k * step (both rounded to 1e-6): drops accumulated float noise."""
    step = round(float(a[-1] - a[0]) / (a.size - 1), 6)
    return round(float(a[0]), 6) + np.arange(a.size) * step


def _raster_coords(d: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Raster lat/lon without float noise, dropping a repeated 180 meridian (-180 is kept)."""
    lat, lon = _regular(d['lat'].values), _regular(d['lon'].values)
    return lat, (lon[:-1] if round(lon[-1] - lon[0], 6) == 360 else lon)


def band_attrs(metric: str, band: str) -> dict[str, str]:
    """long_name/units as in the raw files (the cube is built from a template, so set here)."""
    n = band.split('_')[1]
    if metric == 'return_periods':
        return {'long_name': f'Return period for intensity ≥ {n} m/s', 'units': 'years'}
    return {'long_name': f'Exceedance intensity for RP={n} years', 'units': 'm/s'}


def template(metric: str, rep: Rep, lats: np.ndarray, lons: np.ndarray) -> xr.Dataset:
    shape = (len(SCENARIOS), len(PERIODS), len(VARIANTS), len(GCMS), lats.size, lons.size)
    chunks = (1, -1, -1, -1, -1, -1)  # one dask chunk per scenario, the unit of each commit
    ds = xr.Dataset(
        {
            b: (
                DIMS,
                da.full(shape, np.nan, dtype='float32', chunks=chunks),
                band_attrs(metric, b),
            )
            for b in BANDS[metric]
        },
        coords={
            'scenario': SCENARIOS,
            'period': PERIODS,
            'variant': VARIANTS,
            'gcm': GCMS,
            'lat': lats,
            'lon': lons,
        },
    )
    return _finalize(ds, metric, rep)


def _finalize(ds: xr.Dataset, metric: str, rep: Rep, **attrs: str) -> xr.Dataset:
    """Shared attrs, CRS and GeoZarr metadata for GCM and ERA5 groups."""
    ds.attrs.update(
        {
            'title': f'CHAZ tropical-cyclone {metric} ({rep})',
            'source': 'https://datadryad.org/dataset/doi:10.5061/dryad.qfttdz0vz',
            'source_doi': 'https://doi.org/10.5061/dryad.qfttdz0vz',
            'paper_doi': 'https://doi.org/10.1038/s41597-025-06452-0',
            'license': 'CC0-1.0',
            'representation': rep,
            'bands': BANDS[metric],  # zarr does not keep array order; this is the canonical order
            **attrs,
        }
    )
    ds = ds.proj.assign_crs(spatial_ref='EPSG:4326').proj.write_crs_info()
    return attach_geozarr_metadata(ds, x_dim='lon', y_dim='lat')


def encoding(ds: xr.Dataset) -> dict:
    enc = spacetimer.recommend(ds, 'space')
    # Scenario axis 1 in chunks and shards: each commit writes whole shards, so none are rewritten
    # (and kept as old versions) later.
    for e in enc.values() if 'scenario' in ds.dims else ():  # ERA5 groups are 2-D
        for k in ('chunks', 'shards'):
            if k in e:
                e[k] = (1, *e[k][1:])
    print(spacetimer.report(ds, enc))
    return enc


def assemble(
    cfg: StorageConfig,
    metric: str,
    rep: Rep,
    scenario: str,
    present: set[str],
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[xr.Dataset, int]:
    """All files of one scenario as a (1, period, variant, gcm, lat, lon) block, and the missing count."""
    shape = (1, len(PERIODS), len(VARIANTS), len(GCMS), lats.size, lons.size)
    data = {b: np.full(shape, np.nan, dtype='float32') for b in BANDS[metric]}
    missing = 0
    for p, period in enumerate(PERIODS):
        for v, variant in enumerate(VARIANTS):
            for g, gcm in enumerate(GCMS):
                key = gcm_key(metric, gcm, scenario, period, variant, rep)
                if key not in present:
                    missing += 1
                    continue
                src = read_nc(cfg, key)
                if rep == 'raster' and not all(
                    np.array_equal(a, b) for a, b in zip(_raster_coords(src), (lats, lons))
                ):
                    raise ValueError(f'{key}: raster grid differs from the first file')
                for b in BANDS[metric]:
                    data[b][0, p, v, g] = (
                        _scatter(src, b, lons, lats)
                        if rep == 'points'
                        else src[b].values[:, : lons.size]
                    )
    return xr.Dataset({b: (DIMS, arr) for b, arr in data.items()}), missing


def _committed(repo: icechunk.Repository, group: str) -> set[str] | None:
    """Scenarios committed since the group's latest template, or None if it has no template yet."""
    done = set()
    for snap in repo.ancestry(branch='main'):  # newest first
        if snap.message == f'{group} template':
            return done
        if snap.message.startswith(f'{group} '):
            done.add(snap.message.removeprefix(f'{group} '))
    return None


def run(
    metric: Metric, rep: Rep = 'points', overwrite: bool = False, prefix: str = 'input'
) -> None:
    cfg = StorageConfig()
    store = f'{_check_prefix(prefix)}/{STORE}'  # raw inputs always read from RAW_PREFIX
    group = group_name(metric, rep)
    present = present_keys(cfg, metric, rep)
    if not present:
        raise FileNotFoundError(f'no raw {group} files under {cfg.uri(RAW_PREFIX)}')
    repo = icechunk.Repository.open_or_create(cfg.icechunk_storage(store))
    done = None if overwrite else _committed(repo, group)

    if done is None:
        lats, lons = grid(cfg, sorted(present), rep)
        tmpl = template(metric, rep, lats, lons)
        session = repo.writable_session('main')
        with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore', category=UserWarning
            )  # dask chunks span many zarr chunks
            tmpl.to_zarr(
                session.store,
                group=group,
                mode='w',
                compute=False,
                encoding=encoding(tmpl),
                consolidated=False,
                zarr_format=3,
            )
        session.commit(f'{group} template')
        done = set()
    else:
        existing = xr.open_zarr(
            repo.readonly_session('main').store, group=group, consolidated=False
        )
        lats, lons = existing['lat'].values, existing['lon'].values
        print(f'resuming {group}: done={sorted(done)}')

    for s, scenario in enumerate(SCENARIOS):
        if scenario in done:
            continue
        block, missing = assemble(cfg, metric, rep, scenario, present, lats, lons)
        session = repo.writable_session('main')
        block.to_zarr(
            session.store,
            group=group,
            region={'scenario': slice(s, s + 1)},
            consolidated=False,
            zarr_format=3,
        )
        snapshot = session.commit(f'{group} {scenario}')
        print(f'{group} {scenario}: {missing} missing files NaN-padded -> {snapshot}')
    print(f'wrote {cfg.uri(store)} group {group}')


def run_era5(
    metric: Metric, rep: Rep = 'points', overwrite: bool = False, prefix: str = 'input'
) -> None:
    """ERA5-forced baseline as its own (lat, lon) group on its own grid, in one commit."""
    cfg = StorageConfig()
    store = f'{_check_prefix(prefix)}/{STORE}'
    group, key = group_name(metric, rep, era5=True), era5_key(metric, rep)
    try:
        obstore.head(cfg.object_store(), key)
    except FileNotFoundError as e:
        raise FileNotFoundError(f'no raw ERA5 file at {cfg.uri(key)}') from e
    repo = icechunk.Repository.open_or_create(cfg.icechunk_storage(store))
    msg = f'{group} era5'
    if not overwrite and any(s.message == msg for s in repo.ancestry(branch='main')):
        print(f'{group} already committed; pass --overwrite to rebuild')
        return
    src = read_nc(cfg, key)
    if rep == 'points':
        lats, lons = grid(cfg, [key], rep)
        data = {b: _scatter(src, b, lons, lats) for b in BANDS[metric]}
    else:
        lats, lons = _raster_coords(src)
        # raw raster bands are '<prefix>_<band>' (thr_thr_33, rp_rp_10)
        data = {
            b: src[f'{b.split("_")[0]}_{b}'].values[:, : lons.size].astype('float32')
            for b in BANDS[metric]
        }
    ds = xr.Dataset(
        {b: (('lat', 'lon'), arr, band_attrs(metric, b)) for b, arr in data.items()},
        coords={'lat': lats, 'lon': lons},
    )
    ds = _finalize(ds, metric, rep, forcing='ERA5')
    session = repo.writable_session('main')
    ds.to_zarr(
        session.store,
        group=group,
        mode='w',
        encoding=encoding(ds),
        consolidated=False,
        zarr_format=3,
    )
    print(f'wrote {cfg.uri(store)} group {group} -> {session.commit(msg)}')


def main(
    metric: Annotated[Metric, typer.Option()],
    representation: Rep = 'points',
    source: Literal['gcm', 'era5'] = 'gcm',
    prefix: str = 'input',
    overwrite: bool = False,
) -> None:
    (run_era5 if source == 'era5' else run)(metric, representation, overwrite, prefix)


if __name__ == '__main__':
    typer.run(main)
