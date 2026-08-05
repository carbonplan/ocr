# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1",
#   "numpy",
#   "pyarrow",
#   "s3fs>=2024.0.0",
#   "topozarr==0.1.2",
#   "xarray",
#   "xproj",
#   "zarr>=3",
# ]
# ///
"""Sample the CHAZ damage stores onto CONUS building footprints.

Writes one GeoParquet of Overture building footprints carrying the expected
annual damage bands (`ead`, `ead_lower`, `ead_upper`) from four stores: ERA5
plus the multi-model medians of one scenario/variant across base, fut1 and
fut2. Columns are suffixed by period (`ead_era5`, `ead_base`, ...
`ead_upper_fut2`); scenario, variant and calibration travel in the file name
and the parquet KV metadata, so the schema is the same whichever subset is
built.

Sampling is a nearest-cell lookup at the footprint's bbox centroid, so at the
stores' 300 arcsec (~9 km) grid every building in a cell shares its value —
cell expectations, not building-level estimates. The valid (non-NaN) cells of
each store become in-memory DuckDB join tables; buildings are prefiltered to
the valid cells' bounding box and matched on integer cell indices (the ERA5
grid sits half a cell off the GCM grid, so each store gets its own mapping).
By default a building is kept only if some store's `ead_upper` is positive at
its cell — roughly the eastern half of CONUS; `--keep-zeros` keeps every
building with any sampled value instead. In kept rows NULL means outside a
store's coverage, 0.0 means no TC wind damage.

The scan streams the ~14 GB buildings parquet, so run this in us-west-2 next
to the bucket (`coiled run --region us-west-2 -- ...`); it works anywhere but
pulls the whole file down. Rows are sorted into grid row-major order —
DuckDB's parallel COPY does not keep scan order — so the output row groups
get tight bbox envelopes and bbox-filtered reads prune them; the sort spills,
so give the VM tens of GB of disk headroom. Output lands at
s3://carbonplan-ocr/ocr-explore/CHAZ/buildings/<id>.parquet with a record
merged into a manifest.json alongside.

`verify` re-checks the written file: value ranges, the ead envelope ordering,
KV metadata against the running code, the row-group clustering, and an
independent xarray re-sample of a random subset of rows.

Usage:
  uv run chaz_buildings.py build --scenario ssp370 --variant CRH
  uv run chaz_buildings.py verify --scenario ssp370 --variant CRH
"""

from __future__ import annotations

import argparse
import json
import sys

import duckdb
import numpy as np
import pyarrow as pa
import s3fs
import xarray as xr
from chaz_damage import CAL_TAG, CALIBRATIONS, damage_id
from chaz_matrix import BUCKET, PERIODS, PREFIX, SCENARIOS, VARIANTS, store_uri

BUILDINGS_URI = (
    's3://carbonplan-ocr/input/fire-risk/vector/overture-maps/'
    'CONUS-overture-region-tagged-buildings-2025-09-24.0.parquet'
)
OUT = f'{PREFIX}/CHAZ/buildings'
EAD_BANDS = ['ead', 'ead_lower', 'ead_upper']


def output_id(scenario: str, variant: str, calibration: str) -> str:
    return f'chaz_buildings_ead_conus_{scenario}_{variant}_{CAL_TAG[calibration]}median'


def output_uri(scenario: str, variant: str, calibration: str) -> str:
    return f's3://{BUCKET}/{OUT}/{output_id(scenario, variant, calibration)}.parquet'


def tagged_stores(scenario: str, variant: str, calibration: str) -> dict[str, str]:
    """Column-suffix -> store-id: ERA5 plus the per-period median stores."""
    out = {'era5': damage_id(None, None, None, calibration)}
    for period in PERIODS:
        out[period] = damage_id(scenario, period, variant, calibration, median=True)
    return out


def load_store(sid: str) -> xr.Dataset:
    return xr.open_zarr(f'{store_uri(sid)}/0', consolidated=False).load().squeeze(drop=True)


class Grid:
    """One store's uniform lat/lon grid plus its valid-cell join table."""

    def __init__(self, tag: str, ds: xr.Dataset):
        self.tag = tag
        ead = ds['ead'].values
        if ead.ndim != 2:
            raise ValueError(f'{tag}: expected 2-D bands, got dims {ds["ead"].dims}')
        lat = ds['lat'].values.astype('float64')
        lon = ds['lon'].values.astype('float64')
        self.lat0, self.dlat = float(lat[0]), float((lat[-1] - lat[0]) / (lat.size - 1))
        self.lon0, self.dlon = float(lon[0]), float((lon[-1] - lon[0]) / (lon.size - 1))
        iy, ix = np.nonzero(np.isfinite(ead))
        self.cells = pa.table(
            {
                'iy': iy.astype('int32'),
                'ix': ix.astype('int32'),
                **{f'{b}_{tag}': ds[b].values[iy, ix].astype('float32') for b in EAD_BANDS},
            }
        )
        ys, xs = lat[iy], lon[ix]
        self.bounds = (
            float(xs.min()) - abs(self.dlon) / 2,
            float(ys.min()) - abs(self.dlat) / 2,
            float(xs.max()) + abs(self.dlon) / 2,
            float(ys.max()) + abs(self.dlat) / 2,
        )

    def index_sql(self) -> str:
        return (
            f'CAST(round((yc - ({self.lat0!r})) / ({self.dlat!r})) AS INTEGER) AS iy_{self.tag},\n'
            f'      CAST(round((xc - ({self.lon0!r})) / ({self.dlon!r})) AS INTEGER) AS ix_{self.tag}'
        )


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for ext in ('httpfs', 'aws', 'spatial'):
        con.install_extension(ext)
        con.load_extension(ext)
    con.sql(
        "CREATE OR REPLACE SECRET s3_chain (TYPE s3, PROVIDER credential_chain, REGION 'us-west-2')"
    )
    return con


def value_columns(tags) -> list[str]:
    return [f'{b}_{t}' for t in tags for b in EAD_BANDS]


def _write_manifest(s3: s3fs.S3FileSystem, record: dict) -> None:
    """Merge the record into CHAZ/buildings/manifest.json (keyed by id)."""
    key = f'{BUCKET}/{OUT}/manifest.json'
    existing = {}
    if s3.exists(key):
        with s3.open(key, 'rb') as fh:
            existing = {r['id']: r for r in json.load(fh).get('files', [])}
    existing[record['id']] = record
    with s3.open(key, 'w') as fh:
        json.dump({'files': sorted(existing.values(), key=lambda r: r['id'])}, fh, indent=1)
    print(f'manifest updated: s3://{key}')


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def select_sql(grids: list[Grid], keep_zeros: bool = False) -> str:
    x0 = min(g.bounds[0] for g in grids)
    y0 = min(g.bounds[1] for g in grids)
    x1 = max(g.bounds[2] for g in grids)
    y1 = max(g.bounds[3] for g in grids)
    idx = ',\n      '.join(g.index_sql() for g in grids)
    vals = ',\n    '.join(f'cells_{g.tag}.{b}_{g.tag}' for g in grids for b in EAD_BANDS)
    joins = '\n  '.join(
        f'LEFT JOIN cells_{g.tag}'
        f' ON cells_{g.tag}.iy = s.iy_{g.tag} AND cells_{g.tag}.ix = s.ix_{g.tag}'
        for g in grids
    )
    # a NULL column means outside a store's NA2/data coverage while 0.0 means
    # no TC wind damage there, so zeros stay in the columns — but a row that is
    # zero in every store's widest envelope says nothing an absent row doesn't
    keep = (
        ' OR '.join(f'cells_{g.tag}.ead_{g.tag} IS NOT NULL' for g in grids)
        if keep_zeros
        else ' OR '.join(f'cells_{g.tag}.ead_upper_{g.tag} > 0' for g in grids)
    )
    return f"""
  WITH b AS (
    SELECT
      block_geoid AS GEOID,
      state_abbrev AS state,
      county_name AS county,
      bbox,
      geometry,
      (bbox.xmin + bbox.xmax) / 2 AS xc,
      (bbox.ymin + bbox.ymax) / 2 AS yc
    FROM read_parquet('{BUILDINGS_URI}')
    WHERE bbox.xmin BETWEEN {x0!r} AND {x1!r}
      AND bbox.ymin BETWEEN {y0!r} AND {y1!r}
  ),
  s AS (
    SELECT b.*,
      {idx}
    FROM b
  )
  SELECT
    s.GEOID,
    s.state,
    s.county,
    s.bbox,
    {vals},
    s.geometry
  FROM s
  {joins}
  WHERE {keep}
  ORDER BY s.iy_base, s.ix_base
"""


def cmd_build(args):
    s3 = s3fs.S3FileSystem()
    sid_map = tagged_stores(args.scenario, args.variant, args.calibration)
    uri = output_uri(args.scenario, args.variant, args.calibration)
    if not args.force and s3.exists(uri.removeprefix('s3://')):
        sys.exit(f'{uri} exists; use --force')
    for tag, sid in sid_map.items():
        if not s3.exists(f'{store_uri(sid)}/0/zarr.json'):
            sys.exit(f'source store missing: {sid}')

    grids = []
    for tag, sid in sid_map.items():
        g = Grid(tag, load_store(sid))
        print(f'  {tag}: {g.cells.num_rows:,} valid cells from {sid}')
        grids.append(g)

    con = connect()
    for g in grids:
        con.register(f'cells_{g.tag}', g.cells)

    kv = {
        'scenario': args.scenario,
        'variant': args.variant,
        'calibration': args.calibration,
        'sources': json.dumps(sid_map),
        'buildings_source': BUILDINGS_URI,
        'sampling': 'nearest 300 arcsec cell at the footprint bbox centroid',
        'row_filter': 'any store non-null' if args.keep_zeros else 'any store ead_upper > 0',
    }
    kv_sql = ', '.join(f"'{k}': '{v.replace(chr(39), chr(39) * 2)}'" for k, v in kv.items())
    query = (
        f'COPY ({select_sql(grids, keep_zeros=args.keep_zeros)}) '
        f"TO '{uri}' (FORMAT parquet, COMPRESSION zstd, KV_METADATA {{{kv_sql}}})"
    )
    print(f'sampling buildings -> {uri}')
    con.sql(query)

    cols = value_columns(sid_map)
    counts = con.sql(
        f"SELECT count(*), {', '.join(f'count({c})' for c in cols)} FROM read_parquet('{uri}')"
    ).fetchone()
    non_null = dict(zip(cols, counts[1:]))
    print(f'wrote {counts[0]:,} buildings; non-null: {non_null}')

    _write_manifest(
        s3,
        {
            'id': output_id(args.scenario, args.variant, args.calibration),
            'path': uri,
            'columns': ['GEOID', 'state', 'county', 'bbox', *cols, 'geometry'],
            'rows': counts[0],
            'non_null': non_null,
            **kv,
            'sources': sid_map,
        },
    )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def cmd_verify(args):
    sid_map = tagged_stores(args.scenario, args.variant, args.calibration)
    uri = output_uri(args.scenario, args.variant, args.calibration)
    print(f'verifying {uri}')
    con = connect()
    rel = f"read_parquet('{uri}')"
    failures = 0

    kv = {
        k.decode() if isinstance(k, bytes) else str(k): v.decode()
        if isinstance(v, bytes)
        else str(v)
        for k, v in con.sql(f"SELECT key, value FROM parquet_kv_metadata('{uri}')").fetchall()
    }
    for field, want in (
        ('scenario', args.scenario),
        ('variant', args.variant),
        ('calibration', args.calibration),
        ('sources', json.dumps(sid_map)),
    ):
        if kv.get(field) != want:
            print(f'STALE: metadata {field}={kv.get(field)!r}, this code says {want!r}')
            failures += 1

    n = con.sql(f'SELECT count(*) FROM {rel}').fetchone()[0]
    all_null = con.sql(
        f'SELECT count(*) FROM {rel} WHERE ' + ' AND '.join(f'ead_{t} IS NULL' for t in sid_map)
    ).fetchone()[0]
    print(f'{n:,} rows, {all_null:,} with no value in any store')
    if all_null:
        failures += 1
    for tag in sid_map:
        filled, lo, hi, bad = con.sql(
            f'SELECT count(ead_{tag}), min(ead_lower_{tag}), max(ead_upper_{tag}), '
            f'count(*) FILTER (WHERE ead_lower_{tag} > ead_{tag} + 1e-6 '
            f'OR ead_{tag} > ead_upper_{tag} + 1e-6) FROM {rel}'
        ).fetchone()
        problems = []
        if not filled:
            problems.append('all-NaN')
        elif lo < 0 or hi > 1:
            problems.append(f'outside [0,1] ({lo:.4f}..{hi:.4f})')
        if bad:
            problems.append(f'ead outside envelope at {bad:,} rows')
        failures += len(problems)
        status = '; '.join(problems) if problems else 'ok'
        print(f'  {tag}: {filled:,} filled, ead_lower>={lo:.4f} ead_upper<={hi:.4f}  {status}')

    # bbox-filtered reads prune on row-group stats, so the output must be
    # spatially clustered in at least one dimension (the build's grid-order
    # sort provides it; parallel COPY alone does not)
    meta = con.sql(
        f'SELECT row_group_id, path_in_schema, stats_min_value::DOUBLE, stats_max_value::DOUBLE '
        f"FROM parquet_metadata('{uri}') WHERE path_in_schema IN ('bbox, xmin', 'bbox, ymin')"
    ).fetchall()
    env = {}
    for gid, col, lo, hi in meta:
        env.setdefault(gid, {})[col.removeprefix('bbox, ')] = hi - lo
    spans = [(g['xmin'], g['ymin']) for g in env.values() if 'xmin' in g and 'ymin' in g]
    med_w = float(np.median([w for w, _ in spans]))
    med_h = float(np.median([h for _, h in spans]))
    print(f'{len(spans)} row groups; median bbox-stat envelope {med_w:.2f} x {med_h:.2f} deg')
    if med_w > 5 and med_h > 5:
        print('  poorly clustered: bbox-filtered reads will scan most row groups')
        failures += 1

    cols = value_columns(sid_map)
    sample = con.sql(
        f'SELECT bbox, {", ".join(cols)} FROM {rel} USING SAMPLE {args.n} ROWS'
    ).to_arrow_table()
    bbox = sample['bbox'].to_pylist()
    xc = np.array([(b['xmin'] + b['xmax']) / 2 for b in bbox])
    yc = np.array([(b['ymin'] + b['ymax']) / 2 for b in bbox])
    for tag, sid in sid_map.items():
        ds = load_store(sid)
        pts = {'lat': xr.DataArray(yc, dims='pt'), 'lon': xr.DataArray(xc, dims='pt')}
        for b in EAD_BANDS:
            got = np.array(sample[f'{b}_{tag}'].to_pylist(), dtype='float64')
            want = ds[b].sel(**pts, method='nearest').values.astype('float64')
            differ = int(
                (~(np.isclose(got, want, rtol=0, atol=0) | (np.isnan(got) & np.isnan(want)))).sum()
            )
            # nearest-neighbour ties at cell edges can legitimately disagree
            # between round() and xarray's sel; anything beyond a trace is a bug
            if differ > max(1, args.n // 1000):
                print(f'  {b}_{tag}: {differ}/{len(got)} sampled rows disagree with xarray')
                failures += 1
    print('spot-check against xarray nearest-sampling done')
    sys.exit(1 if failures else 0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    for name, help_ in (
        ('build', 'sample the stores onto buildings'),
        ('verify', 'check a written file'),
    ):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument('--scenario', choices=SCENARIOS, default='ssp370')
        sp.add_argument('--variant', choices=VARIANTS, default='CRH')
        sp.add_argument('--calibration', choices=CALIBRATIONS, default='TDR1.0')
        if name == 'build':
            sp.add_argument('--force', action='store_true')
            sp.add_argument(
                '--keep-zeros',
                action='store_true',
                help='keep buildings whose ead_upper is 0 in every store',
            )
        else:
            sp.add_argument('--n', type=int, default=2000, help='rows to spot-check')
    args = p.parse_args()
    {'build': cmd_build, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
