# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "s3fs>=2024.0.0",
#   "topozarr==0.1.2",
#   "xarray",
#   "xproj",
#   "zarr>=3",
# ]
# ///
"""Build the v2 CONUS TC stores: one pyramid per origin carrying everything a
point query needs.

v1 splits the CHAZ-derived products across stores — damage fractions and ead
in chaz_damage_fraction_conus_*, winds in chaz_exceedance_intensity_*, the
threshold recurrence bands in chaz_return_periods_* — so a viewer showing more
than the headline EAD has to open several stores. v2 joins them on the CONUS
grid, with return period as a real dimension instead of six sibling variables:

  damage_fraction  (return_period, lat, lon)  Eberenz NA2 impact function, 0..1
  wind_speed       (return_period, lat, lon)  1-min sustained 10-m wind, m/s
  ead / ead_lower / ead_upper      (lat, lon) as in chaz_damage.py, yr^-1
  rp_exceed_33 / rp_exceed_50      (lat, lon) years between >=33 / >=50 m/s winds

topozarr chunks non-spatial dims to 1, so each return-period slice is its own
chunk: rendering one band fetches the same bytes as a v1 2D variable, and a
point query pulls the whole damage curve with one array-valued selector.

Only the app-facing origins are built — ERA5 plus the multi-model median per
(scenario, period, variant); the v1 per-GCM stores remain for member-level
exploration. Medians follow v1: every member is transformed first, then the
NaN-aware median is taken per variable, so v2 ead and damage_fraction match
the v1 median stores value-for-value. A consequence worth knowing when the
viewer pairs the bands: median wind_speed and median damage_fraction are each
medians in their own right but are not related through the impact function
(with an even member count the median averages the two middle members, so it
does not commute with f).

The recurrence bands come from the return_periods stores, which are gridded
independently from the same point set, so they are snapped onto the wind grid
(nearest node within half a cell) before masking.

Output: chaz_conus_v2_* pyramids next to the v1 stores in
s3://carbonplan-ocr/ocr-explore/CHAZ/processed/, merged into the shared
manifest.json (18 median + 1 ERA5 = 19 stores per calibration).

Usage:
  uv run chaz_v2.py build --scenario ssp370 --variant CRH
  uv run chaz_v2.py build --all
  uv run chaz_v2.py build --era5-only
  uv run chaz_v2.py verify chaz_conus_v2_ERA5_points
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import s3fs
import xarray as xr
import xproj  # noqa: F401 — registers .proj accessor
import zarr
from chaz_damage import (
    CAL_TAG,
    CALIBRATIONS,
    EAD_BOUNDS,
    EAD_CONVENTION,
    LEVELS,
    REGION,
    V_THRESH,
    VHALF,
    VHALF_EDR,
    WIND_CAP,
    _check_manifest_record,
    _open_conus,
    compute_ead,
    damage_fraction,
    na2_mask,
    source_id,
)
from chaz_matrix import (
    BANDS as MATRIX_BANDS,
    CLIM as MATRIX_CLIM,
    PERIODS,
    SCENARIOS,
    VARIANTS,
    _write_manifest,
    _write_pyramid,
    store_id,
    store_uri,
)

WIND_BANDS = MATRIX_BANDS['exceedance_intensity']  # rp_10 .. rp_1000
THR_BANDS = MATRIX_BANDS['return_periods']  # thr_33, thr_50
RETURN_PERIODS = np.array([int(b.rsplit('_', 1)[1]) for b in WIND_BANDS], dtype='int32')
RP_EXCEED = {'thr_33': 'rp_exceed_33', 'thr_50': 'rp_exceed_50'}

CLIM = {
    'damage_fraction': [0.0, 0.4],
    'wind_speed': MATRIX_CLIM['exceedance_intensity'],
    'rp_exceed': MATRIX_CLIM['return_periods'],
}

ATTRS = {
    'damage_fraction': {
        'long_name': 'TC damage fraction at each return period',
        'units': '1',
    },
    'wind_speed': {
        'long_name': '1-min sustained 10-m wind at each return period',
        'units': 'm s-1',
    },
    'ead': {'long_name': 'expected annual damage at unit exposure', 'units': 'yr-1'},
    'ead_lower': {
        'long_name': 'ead with v_half at the calibration 75th percentile',
        'units': 'yr-1',
    },
    'ead_upper': {
        'long_name': 'ead with v_half at the calibration 25th percentile',
        'units': 'yr-1',
    },
    'rp_exceed_33': {'long_name': 'return period of winds >= 33 m/s', 'units': 'yr'},
    'rp_exceed_50': {'long_name': 'return period of winds >= 50 m/s', 'units': 'yr'},
}


def v2_id(scenario, period, variant, calibration) -> str:
    if scenario is None:
        return f'chaz_conus_v2_ERA5_{CAL_TAG[calibration]}points'
    return f'chaz_conus_v2_{scenario}_{period}_{variant}_median_{CAL_TAG[calibration]}points'


def thr_source_id(scenario, period, variant) -> str:
    if scenario is None:
        return 'chaz_return_periods_ERA5_points'
    return store_id('return_periods', scenario, period, variant, 'points')


def _align(thr: xr.Dataset, wind: xr.Dataset) -> xr.Dataset:
    """Snap the return_periods grid onto the wind grid (same origin, gridded
    independently, so the lattices should coincide; the tolerance turns any
    drift into NaN rather than a silent half-cell shift)."""
    step = float(abs(wind.lat.values[1] - wind.lat.values[0]))
    out = thr.reindex(lat=wind.lat, lon=wind.lon, method='nearest', tolerance=step / 2)
    b = THR_BANDS[0]
    before = int(np.isfinite(thr[b].values).sum())
    after = int(np.isfinite(out[b].values).sum())
    if after < before:
        print(f'    note: {before - after:,} of {before:,} {b} cells fell off the wind grid')
    return out


def _v2_ds(wind: xr.Dataset, thr: xr.Dataset, calibration: str) -> xr.Dataset:
    """The joined dataset for one origin. `wind`/`thr` may carry a gcm dim;
    it rides along ahead of lat/lon and _median_v2 collapses it."""
    v_half = VHALF[calibration][REGION]
    keep = na2_mask(wind.lat.values, wind.lon.values)

    def clip(a):
        return np.where(keep, a, np.float32('nan')).astype('float32')

    spatial = wind[WIND_BANDS[0]].dims
    v = np.stack([wind[b].values for b in WIND_BANDS])
    data_vars = {
        'damage_fraction': (('return_period', *spatial), clip(damage_fraction(v, v_half))),
        'wind_speed': (('return_period', *spatial), clip(v)),
    }
    for name, vh in [
        ('ead', v_half),
        ('ead_lower', VHALF_EDR[0.75][REGION]),
        ('ead_upper', VHALF_EDR[0.25][REGION]),
    ]:
        data_vars[name] = (spatial, clip(compute_ead(wind, vh)))
    for band, name in RP_EXCEED.items():
        data_vars[name] = (spatial, clip(thr[band].values))
    ds = xr.Dataset(data_vars, coords={'return_period': RETURN_PERIODS, **wind.coords})
    ds['return_period'].attrs['units'] = 'yr'
    for name, attrs in ATTRS.items():
        ds[name].attrs.update(attrs)
    if 'gcm' in ds.coords:
        ds['gcm'].attrs = dict(wind['gcm'].attrs)
    return ds.proj.assign_crs(spatial_ref='EPSG:4326')


def _median_v2(stacked: xr.Dataset) -> xr.Dataset:
    """NaN-aware per-variable median over the gcm axis, as in chaz_matrix."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # all-NaN slice -> NaN (expected over ocean)
        for name, da in stacked.data_vars.items():
            axis = da.dims.index('gcm')
            dims = tuple(d for d in da.dims if d != 'gcm')
            out[name] = (dims, np.nanmedian(da.values, axis=axis).astype('float32'), da.attrs)
    ds = xr.Dataset(
        out,
        coords={
            'return_period': stacked['return_period'].values,
            'lat': stacked['lat'].values,
            'lon': stacked['lon'].values,
        },
    )
    ds['return_period'].attrs = dict(stacked['return_period'].attrs)
    return ds.proj.assign_crs(spatial_ref='EPSG:4326')


def _record(sid, origin, calibration, sources, bounds, **extra) -> dict:
    return {
        'id': sid,
        'metric': 'conus_v2',
        'region': 'conus',
        'origin': origin,
        'representation': 'points',
        'dims': {'return_period': [int(t) for t in RETURN_PERIODS]},
        'variables_3d': ['damage_fraction', 'wind_speed'],
        'variables_2d': ['ead', 'ead_lower', 'ead_upper', 'rp_exceed_33', 'rp_exceed_50'],
        'clim': CLIM,
        'impact_function': 'Eberenz et al. 2021 (NHESS), region NA2',
        'extent': (
            'clipped to NA2 land (USA, Canada); Mexican-coastline and '
            'northern-Bahamas cells and open water are masked out'
        ),
        'calibration': calibration,
        'v_half': VHALF[calibration][REGION],
        'v_half_iqr': [VHALF_EDR[0.25][REGION], VHALF_EDR[0.75][REGION]],
        'v_thresh': V_THRESH,
        'ead_convention': EAD_CONVENTION,
        'wind_cap': WIND_CAP,
        'ead_bounds': EAD_BOUNDS,
        'median_note': (
            'median stores transform each member first, then take the NaN-aware '
            'median per variable, so median wind_speed and median damage_fraction '
            'are not related through the impact function'
        ),
        'sources': sources,
        **bounds,
        **extra,
    }


def _build(scenario, period, variant, calibration, s3, force) -> dict | None:
    sid = v2_id(scenario, period, variant, calibration)
    if not force and s3.exists(f'{store_uri(sid)}/zarr.json'):
        print(f'  {sid}: exists; skip (use --force)')
        return None
    wind_src = source_id(scenario, period, variant)
    thr_src = thr_source_id(scenario, period, variant)
    for src in (wind_src, thr_src):
        if not s3.exists(f'{store_uri(src)}/0/zarr.json'):
            print(f'  source missing: {src}')
            return None
    wind = _open_conus(wind_src)
    thr = _align(_open_conus(thr_src), wind)
    ds = _v2_ds(wind, thr, calibration)
    origin = 'ERA5'
    if scenario is not None:
        ds = _median_v2(ds)
        origin = 'median'
    sources = {'wind': wind_src, 'return_periods': thr_src}
    common = (
        {} if scenario is None else {'scenario': scenario, 'period': period, 'variant': variant}
    )
    return _record(
        sid, origin, calibration, sources, _write_pyramid(ds, sid, LEVELS, 'v2'), **common
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def combos(args):
    for sc in [args.scenario] if args.scenario else SCENARIOS:
        for pe in [args.period] if args.period else PERIODS:
            for va in [args.variant] if args.variant else VARIANTS:
                yield sc, pe, va


def cmd_build(args):
    s3 = s3fs.S3FileSystem()
    manifest = []
    if not args.era5_only:
        for sc, pe, va in combos(args):
            rec = _build(sc, pe, va, args.calibration, s3, args.force)
            if rec:
                manifest.append(rec)
    if args.all or args.era5_only or not (args.scenario or args.period or args.variant):
        rec = _build(None, None, None, args.calibration, s3, args.force)
        if rec:
            manifest.append(rec)
    _write_manifest(s3, manifest)


def check_level(arrs: dict[str, np.ndarray]) -> list[str]:
    """Consistency checks for one pyramid level's arrays (see cmd_verify)."""
    problems = []
    dmg, wind = arrs['damage_fraction'], arrs['wind_speed']
    d = dmg[np.isfinite(dmg)]
    if d.size and (d.min() < 0 or d.max() > 1):
        problems.append(f'damage_fraction outside [0,1] ({d.min():.4f}..{d.max():.4f})')
    w = wind[np.isfinite(wind)]
    if w.size and w.min() < 0:
        problems.append(f'wind_speed negative ({w.min():.2f})')
    for name, a, tol in (('damage_fraction', dmg, 1e-6), ('wind_speed', wind, 1e-4)):
        drops = a[:-1] > a[1:] + tol  # non-decreasing along return_period
        if drops.any():
            problems.append(f'{name} shrinks with return period at {int(drops.sum())} cells')
    mask = np.isnan(dmg[0])
    for name in ('damage_fraction', 'wind_speed'):
        problems += [
            f'{name}[{i}] NaN mask differs from damage_fraction[0]'
            for i, s in enumerate(arrs[name])
            if not np.array_equal(np.isnan(s), mask)
        ]
    for name in ('ead', 'ead_lower', 'ead_upper'):
        if not np.array_equal(np.isnan(arrs[name]), mask):
            problems.append(f'{name} NaN mask differs from damage_fraction[0]')
    out = (arrs['ead_lower'] > arrs['ead'] + 1e-6) | (arrs['ead'] > arrs['ead_upper'] + 1e-6)
    if out.any():
        problems.append(f'ead outside envelope at {int(out.sum())} cells')
    for name in RP_EXCEED.values():
        a = arrs[name]
        r = a[np.isfinite(a)]
        if r.size and r.min() <= 0:
            problems.append(f'{name} non-positive ({r.min():.3f})')
        stray = np.isfinite(a) & mask
        if stray.any():
            problems.append(f'{name} has {int(stray.sum())} cells outside the wind mask')
    return problems


def cmd_verify(args):
    uri = store_uri(args.store)
    print(f'verifying {uri}')
    store = zarr.storage.FsspecStore.from_url(uri, read_only=True)
    root = zarr.open_group(store, mode='r')
    print('root attrs:', {k: str(v)[:80] for k, v in dict(root.attrs).items()})
    failures = 0
    for msg in _check_manifest_record(args.store):
        print(f'STALE: {msg}; rebuild with --force')
        failures += 1
    skip = {'lat', 'lon', 'gcm', 'return_period', 'spatial_ref'}
    for level in range(LEVELS):
        sub = zarr.open_group(store, mode='r', path=f'/{level}')
        arrs = {k: np.asarray(sub[k][:]) for k in set(sub.array_keys()) - skip}
        problems = check_level(arrs)
        failures += len(problems)
        finite = arrs['ead'][np.isfinite(arrs['ead'])]
        rng = f'ead<={finite.max():.4f}' if finite.size else 'ead=all-NaN'
        status = '; '.join(problems) if problems else 'ok'
        print(f'/{level}: shape={arrs["damage_fraction"].shape} {rng}  {status}')
    sys.exit(1 if failures else 0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='build matching v2 stores')
    b.add_argument('--scenario', choices=SCENARIOS)
    b.add_argument('--period', choices=PERIODS)
    b.add_argument('--variant', choices=VARIANTS)
    b.add_argument('--calibration', choices=CALIBRATIONS, default='TDR1.0')
    b.add_argument('--all', action='store_true', help='build the whole matrix')
    b.add_argument('--era5-only', action='store_true')
    b.add_argument('--force', action='store_true')

    v = sub.add_parser('verify', help="read a built store's levels")
    v.add_argument('store')

    args = p.parse_args()
    if args.cmd == 'build' and not (
        args.all or args.scenario or args.period or args.variant or args.era5_only
    ):
        sys.exit('build needs --scenario/--period/--variant, --all, or --era5-only')
    {'build': cmd_build, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
