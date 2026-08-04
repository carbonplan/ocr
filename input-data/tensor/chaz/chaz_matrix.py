# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "h5netcdf",
#   "h5py",
#   "numpy",
#   "s3fs>=2024.0.0",
#   "topozarr==0.1.2",
#   "xarray",
#   "xproj",
#   "zarr>=3",
# ]
# ///
"""Build the CHAZ map-viz matrix: GCM-stacked, multi-band Zarr pyramids.

Where pipeline.py builds one pyramid per single (file, variable), this builds the
full CHAZ combination grid as a small number of multi-dimensional stores. It also
writes one NaN-aware, precomputed median store per combination so median map and
query values agree while retaining partial GCM coverage.

Layout: one store per (metric, scenario, period, variant) with dims
(gcm, lat, lon) and every band as a variable, written to
s3://carbonplan-ocr/ocr-explore/CHAZ/processed/<id>/. ERA5 is its own store
(no gcm/scenario/period/variant). The browser picks a GCM by selector index;
the median store has no gcm dimension. The band is picked by variable.

  origins   ERA5 + 6 GCMs (CESM2, CNRM-CM6-1, EC-Earth3, IPSL-CM6A-LR,
            MIROC6, UKESM1-0-LL)
  scenario  ssp245 ssp370 ssp585      period  base fut1 fut2     variant  CRH SD
  bands     return_periods: thr_33 thr_50
            exceedance_intensity: rp_10 rp_25 rp_50 rp_100 rp_250 rp_1000
  => 18 GCM-stacked stores + 18 median stores + 1 ERA5 = 37 / metric

'points' (default) scatters the native coastal point files onto the 1/12 deg
grid (no over-ocean interpolation). 'raster' reads the published gridded product.

Usage:
  uv run chaz_matrix.py list                      # combos + which raw files exist
  uv run chaz_matrix.py build --metric return_periods --scenario ssp245 --period fut1
  uv run chaz_matrix.py build --all               # whole matrix (current --representation)
  uv run chaz_matrix.py verify <store-id>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr
import xproj  # noqa: F401 — registers .proj accessor
import zarr
from topozarr.coarsen import create_pyramid

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

BUCKET = 'carbonplan-ocr'
PREFIX = 'ocr-explore'
RAW = f'{PREFIX}/CHAZ'
OUT = f'{PREFIX}/CHAZ/processed'

GCMS = ['CESM2', 'CNRM-CM6-1', 'EC-Earth3', 'IPSL-CM6A-LR', 'MIROC6', 'UKESM1-0-LL']
SCENARIOS = ['ssp245', 'ssp370', 'ssp585']
PERIODS = ['base', 'fut1', 'fut2']
VARIANTS = ['CRH', 'SD']
METRICS = ['return_periods', 'exceedance_intensity']
BANDS = {
    'return_periods': ['thr_33', 'thr_50'],
    'exceedance_intensity': ['rp_10', 'rp_25', 'rp_50', 'rp_100', 'rp_250', 'rp_1000'],
}
# Suggested display range per metric (carried into the manifest for the app).
CLIM = {'return_periods': [0, 100], 'exceedance_intensity': [17, 70]}

RES = 1.0 / 12.0  # default native CHAZ GCM grid, 300 arcsec (used as fallback)
LEVELS = 5


def _suffix(metric: str, rep: str) -> str:
    return metric if rep == 'points' else f'{metric}_raster'


def _subdir(rep: str) -> str:
    return 'nc' if rep == 'points' else 'raster.nc'


def gcm_key(metric, gcm, scenario, period, variant, rep) -> str:
    return (
        f'{RAW}/{metric}/{_subdir(rep)}/per-GCM/{gcm}/{scenario}/'
        f'TC_global_0300as_CHAZ_{gcm}_{period}_{scenario}_80ens_{variant}_H08_'
        f'{_suffix(metric, rep)}.nc'
    )


def era5_key(metric, rep) -> str:
    return (
        f'{RAW}/{metric}/{_subdir(rep)}/ERA5/TC_global_0300as_CHAZ_ERA5_{_suffix(metric, rep)}.nc'
    )


def store_id(metric, scenario, period, variant, rep) -> str:
    tag = 'points' if rep == 'points' else 'raster'
    return f'chaz_{metric}_{scenario}_{period}_{variant}_{tag}'


def era5_id(metric, rep) -> str:
    return f'chaz_{metric}_ERA5_{"points" if rep == "points" else "raster"}'


def median_id(metric, scenario, period, variant, rep) -> str:
    tag = 'points' if rep == 'points' else 'raster'
    return f'chaz_{metric}_{scenario}_{period}_{variant}_median_{tag}'


def store_uri(sid: str) -> str:
    return f's3://{BUCKET}/{OUT}/{sid}'


# ---------------------------------------------------------------------------
# Read + grid
# ---------------------------------------------------------------------------


def _detect_axis(coords: np.ndarray) -> tuple[float, float]:
    """Infer (step, phase) of a coordinate axis from its point values.

    GCM point files sit on a clean 1/12 deg grid. The ERA5 reanalysis is *also*
    predominantly 1/12 deg, but its minimum coordinate is off-phase: most lats lie
    on a 1/12 grid anchored half a cell (1/24 deg) above lat.min(), with a small
    minority on the other phase. Anchoring the lattice at the raw min (the old
    behaviour) puts the majority of points exactly between two rows, so np.round
    collides ~41% of them and empties ~46% of rows -> horizontal NaN bands.

    So we grid at the *dominant* step anchored to the *dominant* phase: lattice
    nodes fall at (k + phase) * step. The off-phase minority (~0.006% for ERA5)
    snaps to the nearest node (last-write-wins) — sub-grid detail, not banding.
    """
    a = np.unique(np.round(np.asarray(coords, dtype='float64').ravel(), 6))
    d = np.diff(a)
    d = d[d > 1e-6]
    if d.size == 0:
        return RES, 0.0
    vals, counts = np.unique(np.round(d, 6), return_counts=True)
    step = float(vals[int(np.argmax(counts))])  # modal spacing, not the minimum
    inv = round(1.0 / step)
    if inv and abs(1.0 / inv - step) < 1e-4:
        step = 1.0 / inv  # snap to an exact 1/n so phases are clean
    frac = np.round(np.mod(np.asarray(coords, dtype='float64').ravel() / step, 1.0), 3)
    frac[frac > 0.999] = 0.0  # values just under a node wrap to 0
    fv, fc = np.unique(frac, return_counts=True)
    phase = float(fv[int(np.argmax(fc))])  # phase shared by the most points
    return step, phase


def _axis_lattice(coords: np.ndarray, step: float, phase: float) -> np.ndarray:
    """Regular lattice with nodes at (k + phase) * step spanning the data."""
    lo = float(np.min(coords))
    hi = float(np.max(coords))
    k0 = int(np.floor(lo / step - phase))
    k1 = int(np.ceil(hi / step - phase))
    return (np.arange(k0, k1 + 1) + phase) * step


def _res_lbl(lons: np.ndarray, rep: str) -> str:
    if rep != 'points':
        return 'raster'
    return f'1/{1 / (lons[1] - lons[0]):.0f} deg'


def _open(s3: s3fs.S3FileSystem, key: str) -> xr.Dataset:
    with s3.open(key, 'rb') as fh:
        return xr.open_dataset(fh, engine='h5netcdf').load()


def _scatter(ds: xr.Dataset, band: str, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Native point file -> 2D grid (NaN where no point), snapping to the lattice."""
    lon, lat = ds['lon'].values, ds['lat'].values
    lon_res, lat_res = lons[1] - lons[0], lats[1] - lats[0]
    ix = np.clip(np.round((lon - lons[0]) / lon_res).astype(int), 0, lons.size - 1)
    iy = np.clip(np.round((lat - lats[0]) / lat_res).astype(int), 0, lats.size - 1)
    collisions = lon.size - np.unique(iy.astype(np.int64) * lons.size + ix).size
    if collisions:
        print(f'    note: {collisions}/{lon.size} {band} off-phase points snapped')
    grid = np.full((lats.size, lons.size), np.nan, dtype='float32')
    grid[iy, ix] = ds[band].values.astype('float32')
    return grid


def _field(ds: xr.Dataset, band: str, lons, lats, rep: str) -> np.ndarray:
    """One band as a (lat, lon) float32 field on the common grid."""
    if rep == 'points':
        return _scatter(ds, band, lons, lats)
    # raster: already gridded; assume aligned lat/lon (ascending), just take values
    return ds[band].values.astype('float32')


def _grid_from(datasets: list[xr.Dataset], rep: str):
    """Common (lons, lats) lattice covering all member files at the native grid."""
    if rep == 'raster':
        d = datasets[0]
        return d['lon'].values, d['lat'].values
    lon_all = np.concatenate([np.asarray(d['lon'].values).ravel() for d in datasets])
    lat_all = np.concatenate([np.asarray(d['lat'].values).ravel() for d in datasets])
    lon_step, lon_phase = _detect_axis(lon_all)
    lat_step, lat_phase = _detect_axis(lat_all)
    return (
        _axis_lattice(lon_all, lon_step, lon_phase),
        _axis_lattice(lat_all, lat_step, lat_phase),
    )


# ---------------------------------------------------------------------------
# Build a store
# ---------------------------------------------------------------------------


def _assemble(metric, scenario, period, variant, rep, s3) -> tuple[xr.Dataset, list[str]] | None:
    """Read GCM files, scatter every band, stack along gcm at FIXED indices.

    Missing GCMs are NaN-padded (not compacted) so the gcm axis is always the
    full GCMS list in order — the app selects a model by its fixed index, so a
    missing model must not shift the others.
    """
    present = {}
    for gcm in GCMS:
        key = f'{BUCKET}/{gcm_key(metric, gcm, scenario, period, variant, rep)}'
        if s3.exists(key):
            present[gcm] = _open(s3, key)
        else:
            print(f'    missing GCM {gcm} -> NaN-padded (index preserved)')
    if not present:
        print('    no GCM files present; nothing to build')
        return None

    lons, lats = _grid_from(list(present.values()), rep)
    nan = np.full((lats.size, lons.size), np.nan, dtype='float32')
    data_vars = {}
    for band in BANDS[metric]:
        slices = [_field(present[g], band, lons, lats, rep) if g in present else nan for g in GCMS]
        data_vars[band] = (('gcm', 'lat', 'lon'), np.stack(slices, axis=0))
    ds = xr.Dataset(
        data_vars,
        coords={'gcm': np.arange(len(GCMS), dtype='int32'), 'lat': lats, 'lon': lons},
    )
    ds['gcm'].attrs['names'] = list(GCMS)  # fixed order; indices stable
    ds = ds.proj.assign_crs(spatial_ref='EPSG:4326')
    print(
        f'    stacked gcm(fixed 6, present={list(present)}) '
        f'{ds.sizes["lat"]}x{ds.sizes["lon"]} @ {_res_lbl(lons, rep)}'
    )
    return ds, list(GCMS)


def _median_ds(stacked: xr.Dataset) -> xr.Dataset:
    """Per-cell median over the gcm axis, ignoring NaN (union coverage). This is
    the map-ready multi-model median — precomputed because zarr-layer can't
    aggregate over partial-coverage bands client-side (see
    zarr-layer-multiband-discard.md)."""
    import warnings

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # all-NaN slice -> NaN (expected over ocean)
        for band in stacked.data_vars:
            med = np.nanmedian(stacked[band].values, axis=0).astype('float32')
            out[band] = (('lat', 'lon'), med)
    ds = xr.Dataset(
        out,
        coords={'lat': stacked['lat'].values, 'lon': stacked['lon'].values},
    )
    ds = ds.proj.assign_crs(spatial_ref='EPSG:4326')
    return ds


def _assemble_era5(metric, rep, s3) -> xr.Dataset | None:
    key = era5_key(metric, rep)
    if not s3.exists(f'{BUCKET}/{key}'):
        print(f'    ERA5 missing ({key})')
        return None
    d = _open(s3, f'{BUCKET}/{key}')
    lons, lats = _grid_from([d], rep)
    data_vars = {b: (('lat', 'lon'), _field(d, b, lons, lats, rep)) for b in BANDS[metric]}
    ds = xr.Dataset(data_vars, coords={'lat': lats, 'lon': lons})
    ds = ds.proj.assign_crs(spatial_ref='EPSG:4326')
    print(
        f'    ERA5 grid={ds.sizes["lat"]}x{ds.sizes["lon"]} @ {_res_lbl(lons, rep)} '
        f'bands={list(data_vars)}'
    )
    return ds


def _write_pyramid(ds: xr.Dataset, sid: str, levels: int = LEVELS, tag: str = 'matrix') -> dict:
    """Write `ds` as a pyramid to the store `sid`, replacing what is there.

    `aws s3 sync --delete` prunes anything a previous build left behind that
    this one did not write — chunks orphaned by a change in shape or chunking,
    which a reader could otherwise still pick up. It is scoped to this store's
    prefix, so unlike a recursive delete of `target` it cannot reach the rest
    of the bucket if `sid` is ever wrong. The guard below is the other half of
    that: an empty `sid` would make the target the whole processed prefix.
    """
    if not sid or '/' in sid:
        raise ValueError(f'refusing to write to store id {sid!r}')
    print(f'  {sid:<58s}', end='', flush=True)
    t0 = time.time()
    pyramid = create_pyramid(ds, levels=levels, x_dim='lon', y_dim='lat', method='mean')
    with tempfile.TemporaryDirectory(prefix=f'chaz-{tag}-') as tmp:
        local = Path(tmp) / 'store'
        pyramid.as_datatree().to_zarr(
            str(local),
            mode='w',
            encoding=pyramid.encoding,
            consolidated=False,
            zarr_format=3,
        )
        mb = sum(f.stat().st_size for f in local.rglob('*') if f.is_file()) / 1e6
        subprocess.run(
            ['aws', 's3', 'sync', str(local), store_uri(sid), '--delete', '--no-progress'],
            check=True,
        )
        print(f'{mb:6.1f} MB {time.time() - t0:4.0f}s')
    bb = [float(ds.lon.min()), float(ds.lat.min()), float(ds.lon.max()), float(ds.lat.max())]
    return {'bounds': [round(b, 4) for b in bb]}


def build_one(metric, scenario, period, variant, rep, s3, force) -> list[dict]:
    """Build the per-GCM stacked store + the precomputed multi-model median store."""
    sid = store_id(metric, scenario, period, variant, rep)
    mid = median_id(metric, scenario, period, variant, rep)
    need_s = force or not s3.exists(f'{store_uri(sid)}/zarr.json')
    need_m = force or not s3.exists(f'{store_uri(mid)}/zarr.json')
    if not need_s and not need_m:
        print(f'  {sid} (+median) exist; skip (use --force)')
        return []
    out = _assemble(metric, scenario, period, variant, rep, s3)
    if out is None:
        return []
    ds, names = out
    common = {
        'metric': metric,
        'scenario': scenario,
        'period': period,
        'variant': variant,
        'representation': rep,
        'bands': BANDS[metric],
        'clim': CLIM[metric],
    }
    recs = []
    if need_s:
        recs.append(
            {'id': sid, 'origin': 'per-gcm', 'gcms': names, **common, **_write_pyramid(ds, sid)}
        )
    if need_m:
        recs.append(
            {'id': mid, 'origin': 'median', **common, **_write_pyramid(_median_ds(ds), mid)}
        )
    return recs


def build_era5(metric, rep, s3, force) -> dict | None:
    sid = era5_id(metric, rep)
    if not force and s3.exists(f'{store_uri(sid)}/zarr.json'):
        print(f'  {sid}: exists; skip (use --force)')
        return None
    ds = _assemble_era5(metric, rep, s3)
    if ds is None:
        return None
    meta = _write_pyramid(ds, sid)
    return {
        'id': sid,
        'metric': metric,
        'origin': 'ERA5',
        'representation': rep,
        'bands': BANDS[metric],
        'clim': CLIM[metric],
        **meta,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def combos(args):
    metrics = [args.metric] if args.metric else METRICS
    scenarios = [args.scenario] if args.scenario else SCENARIOS
    periods = [args.period] if args.period else PERIODS
    variants = [args.variant] if args.variant else VARIANTS
    for m in metrics:
        for sc in scenarios:
            for pe in periods:
                for va in variants:
                    yield m, sc, pe, va


def cmd_build(args):
    rep = args.representation
    s3 = s3fs.S3FileSystem()
    manifest = []
    if not args.era5_only:
        for m, sc, pe, va in combos(args):
            manifest.extend(build_one(m, sc, pe, va, rep, s3, args.force))
    if args.all or args.metric or args.era5_only:  # ERA5 when sweeping a metric
        for m in [args.metric] if args.metric else METRICS:
            rec = build_era5(m, rep, s3, args.force)
            if rec:
                manifest.append(rec)
    _write_manifest(s3, manifest)


def _write_manifest(s3, new_records):
    """Merge new records into CHAZ/processed/manifest.json (keyed by id)."""
    key = f'{BUCKET}/{OUT}/manifest.json'
    existing = {}
    if s3.exists(key):
        with s3.open(key, 'rb') as fh:
            existing = {r['id']: r for r in json.load(fh).get('stores', [])}
    for r in new_records:
        existing[r['id']] = r
    doc = {'stores': sorted(existing.values(), key=lambda r: r['id'])}
    with s3.open(key, 'w') as fh:
        json.dump(doc, fh, indent=2)
    print(f'\nmanifest: {len(doc["stores"])} stores -> s3://{key}')


def cmd_list(args):
    rep = args.representation
    s3 = s3fs.S3FileSystem()
    print(f'representation={rep}\n')
    for m in METRICS:
        present = sum(
            s3.exists(f'{BUCKET}/{gcm_key(m, g, sc, pe, va, rep)}')
            for g in GCMS
            for sc in SCENARIOS
            for pe in PERIODS
            for va in VARIANTS
        )
        total = len(GCMS) * len(SCENARIOS) * len(PERIODS) * len(VARIANTS)
        e = 'yes' if s3.exists(f'{BUCKET}/{era5_key(m, rep)}') else 'NO'
        print(f'{m}: raw GCM files {present}/{total} present, ERA5={e}')
        combos = len(SCENARIOS) * len(PERIODS) * len(VARIANTS)
        print(f'  -> {combos} GCM-stacked + {combos} median stores + ERA5')


def cmd_verify(args):
    sid = args.store
    uri = store_uri(sid)
    print(f'verifying {uri}')
    store = zarr.storage.FsspecStore.from_url(uri, read_only=True)
    root = zarr.open_group(store, mode='r')
    print('root attrs:', {k: str(v)[:80] for k, v in dict(root.attrs).items()})
    skip = {'lat', 'lon', 'gcm', 'spatial_ref'}
    for level in range(LEVELS):
        sub = zarr.open_group(store, mode='r', path=f'/{level}')
        bands = sorted(set(sub.array_keys()) - skip)
        b0 = sub[bands[0]]
        arr = np.asarray(b0[:])
        d = arr[np.isfinite(arr)]
        rng = f'min={d.min():.2f} max={d.max():.2f}' if d.size else 'all-NaN'
        print(f'/{level}: {bands[0]} shape={b0.shape} chunks={b0.chunks} {rng}  (bands={bands})')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='build matching stores')
    b.add_argument('--metric', choices=METRICS)
    b.add_argument('--scenario', choices=SCENARIOS)
    b.add_argument('--period', choices=PERIODS)
    b.add_argument('--variant', choices=VARIANTS)
    b.add_argument('--representation', choices=['points', 'raster'], default='points')
    b.add_argument('--all', action='store_true', help='build the whole matrix')
    b.add_argument('--era5-only', action='store_true', help='rebuild only ERA5 stores')
    b.add_argument('--force', action='store_true', help='rebuild even if present')

    l = sub.add_parser('list', help='show raw-file presence + planned stores')
    l.add_argument('--representation', choices=['points', 'raster'], default='points')

    v = sub.add_parser('verify', help="read a built store's levels")
    v.add_argument('store')

    args = p.parse_args()
    if args.cmd == 'build':
        if not (
            args.all
            or args.metric
            or args.scenario
            or args.period
            or args.variant
            or args.era5_only
        ):
            sys.exit(
                'build needs a filter (--metric/--scenario/--period/--variant), --all, or --era5-only'
            )
        cmd_build(args)
    elif args.cmd == 'list':
        cmd_list(args)
    elif args.cmd == 'verify':
        cmd_verify(args)


if __name__ == '__main__':
    main()
