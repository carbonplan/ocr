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
"""Build CONUS TC damage-fraction pyramids from the processed CHAZ stores.

Applies the regionally calibrated tropical-cyclone impact function of
Eberenz et al. (2021, https://doi.org/10.5194/nhess-21-393-2021) for region
NA2 (USA & Canada) to the exceedance-intensity stores built by chaz_matrix.py.
Because the function is monotonic in wind speed, the T-year damage fraction at
a cell is f(T-year wind), so each rp_* band maps through f directly. Per-GCM
stores are transformed member-by-member and the median store is the NaN-aware
median of the transformed members (with an even member count the median
averages the two middle values, so it does not commute with f).

Each store also carries `ead`, the expected annual damage (EAD) at unit
exposure (yr^-1): the integral of damage over annual exceedance frequency,
reconstructed from the six return levels under the conventions in
EAD_CONVENTION. It targets the same quantity as CLIMADA's frequency-weighted
average annual impact at unit exposure (and is the analog of the OCR fire
product's RPS metric). Checked against an event-set EAD Meiler computed over
CONUS, six return levels get within ~2% of it, and within 1% in the cells
carrying most of the loss. `ead_lower`/`ead_upper` re-evaluate the
integral with v_half at the calibration's per-event IQR (see EAD_BOUNDS): a
sensitivity envelope over the vulnerability choice alone, not confidence
bounds on the EAD — though that choice is the dominant uncertainty for
absolute TC risk per Meiler et al. 2025
(https://doi.org/10.1126/sciadv.adn4607).

The impact function is the Emanuel (2011) sigmoid as parameterized in CLIMADA
(ImpfTropCyclone.from_emanuel_usa):

    vn  = max(V - v_thresh, 0) / (v_half - v_thresh)    v_thresh = 25.7 m/s
    mdd = vn**3 / (1 + vn**3)

CLIMADA samples that on a 5 m/s grid and linearly interpolates between the
samples, so the function v_half was calibrated through is the piecewise-linear
one, not the sigmoid. `_mdr` reproduces it; evaluating the sigmoid in closed
form instead reads ~5% lower over CONUS and several-fold lower in cells near
v_thresh, where a chord spans the bend.

v_half per region comes from the calibration files shipped with CLIMADA
(climada/data/system/tc_impf_cal_v01_<approach>.csv) — the same per-region
medians ImpfSetTropCyclone.calibrated_regional_vhalf() returns. Vendored below;
`check-calibration` re-derives them from the CSVs.

The NA2 function is the only one applied, so the grid is clipped to NA2 land:
non-NA2 cells inside the CONUS bbox (Mexican coastline, northern Bahamas) are
masked out rather than given the wrong function, as is open water. The CHAZ
maps are 1-min sustained 10-m H08 parametric winds with no terrain-roughness
downscaling, the same convention the calibration used, so the raw maps (not
terrain-downscaled winds) are the consistent input.

Output: chaz_damage_fraction_conus_* topozarr pyramids next to the source
stores in s3://carbonplan-ocr/ocr-explore/CHAZ/processed/, merged into the
shared manifest.json (18 per-GCM + 18 median + 1 ERA5 = 37 stores).

Usage:
  uv run chaz_damage.py check-calibration
  uv run chaz_damage.py build --scenario ssp245 --period base --variant CRH
  uv run chaz_damage.py build --all
  uv run chaz_damage.py verify <store-id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr
import xproj  # noqa: F401 — registers .proj accessor
import zarr
from chaz_matrix import (
    BANDS as MATRIX_BANDS,
    BUCKET,
    OUT,
    PERIODS,
    SCENARIOS,
    VARIANTS,
    _median_ds,
    _write_manifest,
    _write_pyramid,
    store_uri,
)

# ---------------------------------------------------------------------------
# Impact function (Eberenz et al. 2021 / CLIMADA)
# ---------------------------------------------------------------------------

V_THRESH = 25.7  # m/s, held constant across regions (Emanuel 2011)

# Per-region median v_half (m/s) from CLIMADA's calibration result files —
# equal to ImpfSetTropCyclone.calibrated_regional_vhalf(approach) with q=0.5.
VHALF = {
    'TDR1.0': {
        'GLB': 110.1,
        'NA1': 66.3,
        'NA2': 89.2,
        'NI': 70.8,
        'OC': 64.1,
        'SI': 52.4,
        'WP1': 66.4,
        'WP2': 188.4,
        'WP3': 112.8,
        'WP4': 190.5,
    },
    'TDR1.5': {
        'GLB': 98.9,
        'NA1': 58.8,
        'NA2': 80.5,
        'NI': 63.7,
        'OC': 56.8,
        'SI': 48.5,
        'WP1': 60.7,
        'WP2': 167.5,
        'WP3': 101.5,
        'WP4': 169.6,
    },
    'RMSF': {
        'GLB': 73.4,
        'NA1': 59.6,
        'NA2': 86.0,
        'NI': 58.7,
        'OC': 49.7,
        'SI': 46.8,
        'WP1': 56.7,
        'WP2': 84.7,
        'WP3': 80.2,
        'WP4': 135.6,
    },
}
CALIBRATIONS = list(VHALF)
REGION = 'NA2'  # USA & Canada

# Interquartile range of the per-event v_half fits (EDR calibration file),
# the vulnerability-uncertainty envelope sampled by Meiler et al. 2025
# (https://doi.org/10.1126/sciadv.adn4607, Table 1) — the dominant source of
# uncertainty for absolute TC risk. Lower v_half = more damage.
VHALF_EDR = {
    0.25: {
        'GLB': 45.1,
        'NA1': 42.7,
        'NA2': 74.05,
        'NI': 44.35,
        'OC': 36.775,
        'SI': 32.5,
        'WP1': 42.8,
        'WP2': 50.55,
        'WP3': 53.8,
        'WP4': 100.175,
    },
    0.75: {
        'GLB': 130.6,
        'NA1': 83.5,
        'NA2': 115.1,
        'NI': 80.9,
        'OC': 87.075,
        'SI': 58.5,
        'WP1': 77.6,
        'WP2': 188.1,
        'WP3': 130.9,
        'WP4': 282.75,
    },
}

EAD_BOUNDS = (
    'ead_lower/ead_upper evaluate the same integral with v_half at the 75th/'
    '25th percentile of the per-event calibration fits (EDR IQR), the '
    'vulnerability envelope of Meiler et al. 2025 (10.1126/sciadv.adn4607)'
)

CLIMADA_CSV = (
    'https://raw.githubusercontent.com/CLIMADA-project/climada_python/main/'
    'climada/data/system/tc_impf_cal_v01_{approach}.csv'
)


IMPF_NODES = (0.0, 120.0, 5.0)  # from_emanuel_usa's np.arange(0, 121, 5)


def _mdd(v: np.ndarray, v_half) -> np.ndarray:
    """The Emanuel sigmoid in closed form."""
    vn = np.maximum(v - V_THRESH, 0.0) / (v_half - V_THRESH)
    vn3 = vn**3
    return vn3 / (1.0 + vn3)


def _mdr(v: np.ndarray, v_half) -> np.ndarray:
    """The sigmoid as CLIMADA evaluates it: sampled on IMPF_NODES by
    from_emanuel_usa, then linearly interpolated by ImpactFunc.calc_mdr. The
    chords sit above the convex curve between nodes, most of all on 25-30
    where v_thresh falls mid-segment, so this reads higher than _mdd. It is
    the function Eberenz et al. calibrated v_half through, so it is the one
    that reproduces published CLIMADA damages.
    """
    lo, hi, step = IMPF_NODES
    x = np.clip(v, lo, hi)
    left = np.clip(np.floor(x / step) * step, lo, hi - step)
    f = (x - left) / step
    return _mdd(left, v_half) * (1.0 - f) + _mdd(left + step, v_half) * f


def damage_fraction(v: np.ndarray, v_half) -> np.ndarray:
    """Mean damage degree (0..1) for wind speed v (m/s); NaN passes through."""
    return _mdr(v, v_half).astype('float32')


RP_YEARS = np.array([10.0, 25.0, 50.0, 100.0, 250.0, 1000.0])  # BANDS order
EAD_LOGT_GRID = np.linspace(0.0, 4.5, 91)  # T = 1..~32,000 yr, dense in log10(T)

# Ceiling on the extrapolated wind, from two bounds. It must clear the largest
# rp_1000 wind in any CHAZ store over NA2 (81.9 m/s) so it never touches a
# measured value, and it should sit near the Atlantic record for 1-min
# sustained wind (~85 m/s), which these ~9 km cell means should stay under.
# That leaves a range, not a number; 90 is a choice inside it, and the
# walkthrough (section 5) scores 85 and 100 to show what the choice is worth.
# Held constant rather than derived per store: a per-store maximum would raise
# the ceiling exactly where the hazard is highest, which is the fragility this
# is meant to remove. Capping here is cheap because the damage curve carries
# little information this far out either — 90 m/s is already NA2's v_half, and
# Eberenz fit that sigmoid on events far below it. That is a reason the cap
# costs nothing, not a reason 90 is the right number.
WIND_CAP = 90.0  # m/s

EAD_CONVENTION = (
    'damage integrated over annual exceedance frequency (1/T) with wind '
    'piecewise-linear in log10(T) between the six rp bands, extended below '
    'rp_10 on the (rp_10, rp_25) slope and beyond rp_1000 on the '
    f'(rp_250, rp_1000) slope and capped at {WIND_CAP:g} m/s, evaluated through '
    'the impact function on a 91-point log10(T) grid from T=1 to ~32,000 yr (zero '
    'once wind falls below v_thresh); damage held constant beyond the grid'
)


def ead_from_levels(v: np.ndarray, v_half) -> np.ndarray:
    """EAD at unit exposure (yr^-1) from a (6, ...) stack of return-level winds
    in BANDS order, per EAD_CONVENTION. `v_half` is a scalar or an array
    broadcastable against `v[0]`, so a per-cell impact function works.
    """
    knots_logt = np.log10(RP_YEARS)
    slope = np.maximum(v[1] - v[0], 0.0) / (knots_logt[1] - knots_logt[0])
    knots_logt = np.concatenate([[0.0], knots_logt])
    knots_v = np.concatenate([(v[0] - slope)[None], v])
    # clipping to the end segments makes the two outer knots extrapolate:
    # below T=10 on the (rp_10, rp_25) slope, past rp_1000 on (rp_250, rp_1000),
    # the latter bounded by WIND_CAP
    seg = np.clip(
        np.searchsorted(knots_logt, EAD_LOGT_GRID, side='right') - 1,
        0,
        len(knots_logt) - 2,
    )
    d = []
    for t, k in zip(EAD_LOGT_GRID, seg):
        w = (t - knots_logt[k]) / (knots_logt[k + 1] - knots_logt[k])
        wind = knots_v[k] * (1.0 - w) + knots_v[k + 1] * w
        d.append(_mdr(np.minimum(wind, WIND_CAP), v_half))
    lam = 1.0 / 10.0**EAD_LOGT_GRID
    ead = lam[-1] * d[-1]  # constant damage past the end of the grid
    for j in range(len(lam) - 1):
        ead += (lam[j] - lam[j + 1]) * 0.5 * (d[j] + d[j + 1])
    return ead


def compute_ead(wind: xr.Dataset, v_half: float) -> np.ndarray:
    """Expected annual damage at unit exposure (yr^-1): integral of damage
    over exceedance frequency, reconstructed from the six return levels per
    EAD_CONVENTION — an approximation of the quantity CLIMADA's
    frequency-weighted average annual impact computes from a full event set,
    within ~2% of one over CONUS (see the walkthrough notebook, Part 4).
    """
    v = np.stack([wind[b].values.astype('float64') for b in BANDS])
    hottest = np.nanmax(v) if np.isfinite(v).any() else 0.0
    if hottest >= WIND_CAP:
        print(
            f'    WARNING: a measured wind reaches {hottest:.1f} m/s, at or above '
            f'WIND_CAP={WIND_CAP}; the ceiling is clipping data, not just '
            f'extrapolation. Raise it or switch to a per-cell bound.'
        )
    return ead_from_levels(v, v_half).astype('float32')


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

CONUS_BBOX = (-125.0, 24.0, -66.0, 50.5)  # lon_min, lat_min, lon_max, lat_max
BANDS = MATRIX_BANDS['exceedance_intensity']  # rp_10 .. rp_1000
CLIM = [0.0, 0.4]  # display range: f(80 m/s, NA2 TDR) ~ 0.39
LEVELS = 4  # CONUS level 0 is ~708x318; the global stores' 5 would over-coarsen

# Store-id fragment per calibration; the CLIMADA default (TDR1.0) is unmarked.
CAL_TAG = {'TDR1.0': '', 'TDR1.5': 'tdr1p5_', 'RMSF': 'rmsf_'}


def source_id(scenario: str | None, period: str | None, variant: str | None) -> str:
    if scenario is None:
        return 'chaz_exceedance_intensity_ERA5_points'
    return f'chaz_exceedance_intensity_{scenario}_{period}_{variant}_points'


def damage_id(scenario, period, variant, calibration, median=False) -> str:
    mid = 'median_' if median else ''
    if scenario is None:
        return f'chaz_damage_fraction_conus_ERA5_{CAL_TAG[calibration]}points'
    return (
        f'chaz_damage_fraction_conus_{scenario}_{period}_{variant}_'
        f'{mid}{CAL_TAG[calibration]}points'
    )


# ---------------------------------------------------------------------------
# NA2 mask
# ---------------------------------------------------------------------------

NA2_ISO = ('USA', 'CAN')  # the countries region NA2 covers
MASK_NPZ = Path(__file__).with_name('na2_mask.npz')
GEOM_NPZ = Path(__file__).with_name('na2_geometry.npz')

# Pinned, not a branch: the mask is committed, and a moving upstream would let
# a rebuild silently disagree with it. `fetch-geometry` re-vendors GEOM_NPZ.
NE_VERSION = 'v5.1.2'
NE_COUNTRIES = (
    f'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
    f'{NE_VERSION}/geojson/ne_50m_admin_0_countries.geojson'
)


def fetch_na2_geometry(path: Path = GEOM_NPZ) -> None:
    """Vendor the NA2 country outlines from Natural Earth into GEOM_NPZ.

    Only needs re-running to move to a new Natural Earth release. Rings are
    stored flat (float32 is ~1 m at these latitudes, far under a 9 km cell)
    with per-ring and per-polygon lengths to rebuild the nesting.
    """
    import json
    import urllib.request

    with urllib.request.urlopen(NE_COUNTRIES) as f:
        countries = json.load(f)
    polys = []
    for feat in countries['features']:
        if feat['properties'].get('ADM0_A3') not in NA2_ISO:
            continue
        geom = feat['geometry']
        parts = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
        polys += [[np.asarray(r, 'float32') for r in part] for part in parts]
    np.savez_compressed(
        path,
        xy=np.concatenate([r for p in polys for r in p]),
        ring_len=np.array([len(r) for p in polys for r in p], 'int32'),
        poly_len=np.array([len(p) for p in polys], 'int32'),
        source=np.array(NE_COUNTRIES),
    )
    n = sum(len(r) for p in polys for r in p)
    print(f'{len(polys)} polygons, {n:,} vertices from {NE_VERSION} -> {path}')


def _na2_polygons() -> list[list[np.ndarray]]:
    """The vendored outlines, as a list of [outer_ring, *holes]."""
    z = np.load(GEOM_NPZ)
    rings = np.split(z['xy'], np.cumsum(z['ring_len'])[:-1])
    out, i = [], 0
    for n in z['poly_len']:
        out.append(rings[i : i + n])
        i += n
    return out


def _grid_key(lat: np.ndarray, lon: np.ndarray) -> str:
    """Cache key for one grid. The ERA5 store sits half a cell off the GCM
    stores (318x708 vs 319x709), so the cache holds a mask per grid."""
    return f'{lat.size}x{lon.size}@{lat[0]:.5f},{lon[0]:.5f}'


def build_na2_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Rasterize the NA2 outlines onto a store grid.

    `all_touched` gives us the rule we want: a cell is kept if any part of it is
    USA or Canada, not just its centre. Biasing that way keeps the coastline and
    the land border whole — at 9.3 km a cell is wider than the Rio Grande, and
    dropping the ones that straddle it would erode the Texas coast.

    What that leaves out is cells wholly inside another country — the Mexican
    coastline and northern Bahamas, which the paper puts in NA1 and which this
    pipeline would otherwise give the NA2 impact function — and open water,
    where a unit-exposure damage fraction means nothing anyway.

    rasterio and shapely are imported here rather than at module scope so that
    reading a cached mask needs neither.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry import Polygon

    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    ascending = lat[-1] > lat[0]
    top = (lat[-1] if ascending else lat[0]) + dlat / 2
    grid = rasterize(
        (Polygon(p[0], p[1:]) for p in _na2_polygons()),
        out_shape=(lat.size, lon.size),
        transform=from_origin(lon[0] - dlon / 2, top, dlon, dlat),
        all_touched=True,
        fill=0,
        default_value=1,
    ).astype(bool)
    return grid[::-1] if ascending else grid  # rasterize is north-up


def na2_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """The NA2 mask for this grid, from the cache or built and added to it."""
    key = _grid_key(lat, lon)
    cached = dict(np.load(MASK_NPZ)) if MASK_NPZ.exists() else {}
    if key in cached:
        return cached[key]
    keep = build_na2_mask(lat, lon)
    print(f'    {keep.sum():,} of {keep.size:,} cells on NA2 land for {key} -> {MASK_NPZ}')
    np.savez_compressed(MASK_NPZ, **cached, **{key: keep})
    return keep


def _open_conus(sid: str) -> xr.Dataset:
    """Level 0 (native 300 arcsec grid) of a processed store, cropped to CONUS."""
    ds = xr.open_zarr(f'{store_uri(sid)}/0', consolidated=False)
    lon0, lat0, lon1, lat1 = CONUS_BBOX
    lat_slice = slice(lat0, lat1) if ds.lat[0] < ds.lat[-1] else slice(lat1, lat0)
    ds = ds.sel(lon=slice(lon0, lon1), lat=lat_slice).load()
    return ds.drop_vars('spatial_ref', errors='ignore')


def _damage_ds(wind: xr.Dataset, calibration: str) -> xr.Dataset:
    v_half = VHALF[calibration][REGION]
    keep = na2_mask(wind.lat.values, wind.lon.values)

    def clip(a):
        return np.where(keep, a, np.float32('nan')).astype('float32')

    data_vars = {b: (wind[b].dims, clip(damage_fraction(wind[b].values, v_half))) for b in BANDS}
    dims = wind[BANDS[0]].dims
    for name, vh in [
        ('ead', v_half),
        ('ead_lower', VHALF_EDR[0.75][REGION]),
        ('ead_upper', VHALF_EDR[0.25][REGION]),
    ]:
        data_vars[name] = (dims, clip(compute_ead(wind, vh)))
    ds = xr.Dataset(data_vars, coords=wind.coords)
    if 'gcm' in ds.coords:
        ds['gcm'].attrs = dict(wind['gcm'].attrs)
    return ds.proj.assign_crs(spatial_ref='EPSG:4326')


def _record(sid, origin, calibration, src, bounds, **extra) -> dict:
    return {
        'id': sid,
        'metric': 'damage_fraction',
        'region': 'conus',
        'origin': origin,
        'representation': 'points',
        'bands': [*BANDS, 'ead', 'ead_lower', 'ead_upper'],
        'clim': CLIM,
        'impact_function': 'Eberenz et al. 2021 (NHESS), region NA2',
        'impact_function_eval': (
            "sampled every 5 m/s and linearly interpolated, as CLIMADA's "
            'from_emanuel_usa + ImpactFunc.calc_mdr do — the form v_half was '
            'calibrated through'
        ),
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
        'source': src,
        **bounds,
        **extra,
    }


def build_one(scenario, period, variant, calibration, s3, force) -> list[dict]:
    """Per-GCM damage store + its multi-model median store for one combo."""
    sid = damage_id(scenario, period, variant, calibration)
    mid = damage_id(scenario, period, variant, calibration, median=True)
    need_s = force or not s3.exists(f'{store_uri(sid)}/zarr.json')
    need_m = force or not s3.exists(f'{store_uri(mid)}/zarr.json')
    if not need_s and not need_m:
        print(f'  {sid} (+median) exist; skip (use --force)')
        return []
    src = source_id(scenario, period, variant)
    if not s3.exists(f'{store_uri(src)}/0/zarr.json'):
        print(f'  source missing: {src}')
        return []
    wind = _open_conus(src)
    dmg = _damage_ds(wind, calibration)
    common = {'scenario': scenario, 'period': period, 'variant': variant}
    recs = []
    if need_s:
        gcms = list(dmg['gcm'].attrs.get('names', []))
        recs.append(
            _record(
                sid,
                'per-gcm',
                calibration,
                src,
                _write_pyramid(dmg, sid, LEVELS, 'damage'),
                gcms=gcms,
                **common,
            )
        )
    if need_m:
        med = _median_ds(dmg)
        recs.append(
            _record(
                mid,
                'median',
                calibration,
                src,
                _write_pyramid(med, mid, LEVELS, 'damage'),
                **common,
            )
        )
    return recs


def build_era5(calibration, s3, force) -> dict | None:
    sid = damage_id(None, None, None, calibration)
    if not force and s3.exists(f'{store_uri(sid)}/zarr.json'):
        print(f'  {sid}: exists; skip (use --force)')
        return None
    src = source_id(None, None, None)
    if not s3.exists(f'{store_uri(src)}/0/zarr.json'):
        print(f'  source missing: {src}')
        return None
    dmg = _damage_ds(_open_conus(src), calibration)
    return _record(sid, 'ERA5', calibration, src, _write_pyramid(dmg, sid, LEVELS, 'damage'))


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
            manifest.extend(build_one(sc, pe, va, args.calibration, s3, args.force))
    if args.all or args.era5_only or not (args.scenario or args.period or args.variant):
        rec = build_era5(args.calibration, s3, args.force)
        if rec:
            manifest.append(rec)
    _write_manifest(s3, manifest)


def _check_manifest_record(sid: str) -> list[str]:
    """Compare the store's manifest record against the running code.

    The band values alone cannot show which conventions produced them, so a
    store built before a convention changed looks perfectly self-consistent.
    This is what catches it — the build prints warnings, but a build's stdout
    is easy to lose.
    """
    key = f'{BUCKET}/{OUT}/manifest.json'
    s3 = s3fs.S3FileSystem()
    if not s3.exists(key):
        return ['no manifest.json alongside the stores']
    with s3.open(key, 'rb') as fh:
        rec = {r['id']: r for r in json.load(fh).get('stores', [])}.get(sid)
    if rec is None:
        return [f'{sid} is not in manifest.json']
    stale = []
    for field, want in (('ead_convention', EAD_CONVENTION), ('wind_cap', WIND_CAP)):
        got = rec.get(field)
        if got is None:
            stale.append(f'manifest record predates `{field}`')
        elif got != want:
            stale.append(f'manifest `{field}` differs from this code ({got!r})')
    return stale


def cmd_verify(args):
    uri = store_uri(args.store)
    print(f'verifying {uri}')
    store = zarr.storage.FsspecStore.from_url(uri, read_only=True)
    root = zarr.open_group(store, mode='r')
    print('root attrs:', {k: str(v)[:80] for k, v in dict(root.attrs).items()})
    skip = {'lat', 'lon', 'gcm', 'spatial_ref'}
    failures = 0
    for msg in _check_manifest_record(args.store):
        print(f'STALE: {msg}; rebuild with --force')
        failures += 1
    for level in range(LEVELS):
        sub = zarr.open_group(store, mode='r', path=f'/{level}')
        bands = sorted(set(sub.array_keys()) - skip)
        arrs = {b: np.asarray(sub[b][:]) for b in bands}
        problems, ranges = [], []
        for b, a in arrs.items():
            d = a[np.isfinite(a)]
            ranges.append(f'{b}<={d.max():.3f}' if d.size else f'{b}=all-NaN')
            if d.size and (d.min() < 0 or d.max() > 1):
                problems.append(f'{b} outside [0,1] ({d.min():.4f}..{d.max():.4f})')
        mask = np.isnan(arrs[bands[0]])
        problems += [
            f'{b} NaN mask differs from {bands[0]}'
            for b in bands[1:]
            if not np.array_equal(np.isnan(arrs[b]), mask)
        ]
        # damage must grow with return period, and mean-coarsening preserves that
        rp = [b for b in BANDS if b in arrs]
        problems += [
            f'{lo} > {hi} at {int((arrs[lo] > arrs[hi] + 1e-6).sum())} cells'
            for lo, hi in zip(rp, rp[1:])
            if (arrs[lo] > arrs[hi] + 1e-6).any()
        ]
        if {'ead', 'ead_lower', 'ead_upper'} <= set(bands):
            out = (arrs['ead_lower'] > arrs['ead'] + 1e-6) | (
                arrs['ead'] > arrs['ead_upper'] + 1e-6
            )
            if out.any():
                problems.append(f'ead outside envelope at {int(out.sum())} cells')
        failures += len(problems)
        shape = arrs[bands[0]].shape
        status = '; '.join(problems) if problems else 'ok'
        print(f'/{level}: shape={shape} {" ".join(ranges)}  {status}')
    sys.exit(1 if failures else 0)


def cmd_check_calibration(_args):
    """Re-derive the vendored v_half medians from CLIMADA's calibration CSVs."""
    import csv
    import io
    import urllib.request

    ok = True

    def check(label, rows, region, q, want):
        nonlocal ok
        vals = np.array(
            [float(r['v_half']) for r in rows if r['cal_region2'] == region]
            if region != 'GLB' or label != 'EDR'
            else [float(r['v_half']) for r in rows]
        )
        got = round(float(np.quantile(vals, q)), 5)
        mark = 'ok' if got == want else f'MISMATCH (csv says {got})'
        if got != want:
            ok = False
        print(f'  {label:7s} {region:4s} q={q:<5} v_half={want:<9} {mark}')

    for approach, expected in VHALF.items():
        url = CLIMADA_CSV.format(approach=approach)
        txt = urllib.request.urlopen(url).read().decode('ISO-8859-1')
        rows = list(csv.DictReader(io.StringIO(txt)))
        for region, want in expected.items():
            check(approach, rows, region, 0.5, want)
    txt = urllib.request.urlopen(CLIMADA_CSV.format(approach='EDR')).read().decode('ISO-8859-1')
    rows = list(csv.DictReader(io.StringIO(txt)))
    for q, expected in VHALF_EDR.items():
        for region, want in expected.items():
            check('EDR', rows, region, q, want)
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='build matching damage stores')
    b.add_argument('--scenario', choices=SCENARIOS)
    b.add_argument('--period', choices=PERIODS)
    b.add_argument('--variant', choices=VARIANTS)
    b.add_argument('--calibration', choices=CALIBRATIONS, default='TDR1.0')
    b.add_argument('--all', action='store_true', help='build the whole matrix')
    b.add_argument('--era5-only', action='store_true')
    b.add_argument('--force', action='store_true')

    v = sub.add_parser('verify', help="read a built store's levels")
    v.add_argument('store')

    sub.add_parser('check-calibration', help='verify vendored v_half against CLIMADA CSVs')
    sub.add_parser('fetch-geometry', help=f're-vendor NA2 outlines from Natural Earth {NE_VERSION}')

    args = p.parse_args()
    if args.cmd == 'build' and not (
        args.all or args.scenario or args.period or args.variant or args.era5_only
    ):
        sys.exit('build needs --scenario/--period/--variant, --all, or --era5-only')
    {
        'build': cmd_build,
        'verify': cmd_verify,
        'check-calibration': cmd_check_calibration,
        'fetch-geometry': lambda _: fetch_na2_geometry(),
    }[args.cmd](args)


if __name__ == '__main__':
    main()
