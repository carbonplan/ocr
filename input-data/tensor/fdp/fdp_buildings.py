# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1",
#   "numpy",
#   "pyarrow",
#   "pyproj",
#   "s3fs>=2024.0.0",
#   "xarray",
#   "zarr>=3",
# ]
# ///
"""Sample the flood damage probability store onto CONUS building footprints.

Writes one GeoParquet of Overture building footprints carrying `fdp`, the
predicted probability of flood damage at the footprint's location.

Sampling is a nearest-cell lookup at the footprint's bbox centroid. The store is
a 100 m grid, close enough to footprint scale that these read as per-building
values rather than the cell expectations a coarse grid would give. Buildings
larger than a cell still take a single value, and neighbours within the same
100 m cell share one.

The grid is in Albers metres while the footprints are in lon/lat, so centroids
are reprojected (EPSG:4326 -> EPSG:5070) before the cell arithmetic. DuckDB's
ST_Transform agrees with pyproj to well under a metre here.

The occupied cells of the store become an in-memory DuckDB join table: the full
grid is 1.5e9 cells, far too many to join against, but the cells that actually
contain a building are a small fraction of that. Buildings outside the store's
valid data are dropped by the join, so every written row has a value.

The scan streams the ~14 GB buildings parquet, so run this in us-west-2 next to
the bucket (see the README); it works anywhere but pulls the whole file down.
Rows are sorted into grid row-major order, since DuckDB's parallel COPY does not
keep scan order, so the output row groups get tight bbox envelopes and
bbox-filtered reads prune them. The sort spills, so give the VM tens of GB of
disk headroom.

Usage:
  python fdp_buildings.py build
  python fdp_buildings.py verify
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
from fdp import BUCKET, CRS, NAME, PREFIX, VARIABLE, store_uri
from pyproj import Transformer

BUILDINGS_URI = (
    's3://carbonplan-ocr/input/fire-risk/vector/overture-maps/'
    'CONUS-overture-region-tagged-buildings-2025-09-24.0.parquet'
)
OUT = f'{PREFIX}/buildings'
OUTPUT_ID = 'fdp_buildings_conus'


def output_uri() -> str:
    return f's3://{BUCKET}/{OUT}/{OUTPUT_ID}.parquet'


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for ext in ('httpfs', 'aws', 'spatial'):
        con.install_extension(ext)
        con.load_extension(ext)
    con.sql(
        "CREATE OR REPLACE SECRET s3_chain (TYPE s3, PROVIDER credential_chain, REGION 'us-west-2')"
    )
    return con


class Grid:
    """The store's native Albers grid, plus the lon/lat window that covers it."""

    def __init__(self, ds: xr.Dataset):
        x = ds['x'].values.astype('float64')
        y = ds['y'].values.astype('float64')
        self.x0, self.dx = float(x[0]), float((x[-1] - x[0]) / (x.size - 1))
        self.y0, self.dy = float(y[0]), float((y[-1] - y[0]) / (y.size - 1))
        self.values = ds[VARIABLE].values
        self.ny, self.nx = self.values.shape

        # Albers is not axis-aligned with lon/lat, so walk the grid's boundary
        # and take the lon/lat envelope of it. A superset is fine: it only
        # prefilters the scan, and the join drops whatever falls outside.
        inv = Transformer.from_crs(CRS, 'EPSG:4326', always_xy=True)
        ex, ey = x[[0, -1]], y[[0, -1]]
        bx = np.concatenate([x, x, np.full(y.size, ex[0]), np.full(y.size, ex[1])])
        by = np.concatenate([np.full(x.size, ey[0]), np.full(x.size, ey[1]), y, y])
        lon, lat = inv.transform(bx, by)
        self.lon0, self.lon1 = float(np.nanmin(lon)) - 0.01, float(np.nanmax(lon)) + 0.01
        self.lat0, self.lat1 = float(np.nanmin(lat)) - 0.01, float(np.nanmax(lat)) + 0.01

    def cells_at(self, iy: np.ndarray, ix: np.ndarray) -> pa.Table:
        """Look up `fdp` at the given indices, keeping only in-grid finite cells."""
        ok = (iy >= 0) & (iy < self.ny) & (ix >= 0) & (ix < self.nx)
        iy, ix = iy[ok], ix[ok]
        v = self.values[iy, ix]
        ok = np.isfinite(v)
        return pa.table(
            {
                'iy': iy[ok].astype('int32'),
                'ix': ix[ok].astype('int32'),
                VARIABLE: v[ok].astype('float32'),
            }
        )


def _indexed_sql(grid: Grid, columns: str) -> str:
    """Buildings in the grid's lon/lat window, with their Albers cell indices."""
    return f"""
  WITH b AS (
    SELECT {columns},
      ST_Transform(
        ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2),
        'EPSG:4326', '{CRS}', always_xy := true
      ) AS g
    FROM read_parquet('{BUILDINGS_URI}')
    WHERE bbox.xmin BETWEEN {grid.lon0!r} AND {grid.lon1!r}
      AND bbox.ymin BETWEEN {grid.lat0!r} AND {grid.lat1!r}
  )
  SELECT b.* EXCLUDE (g),
    CAST(round((ST_Y(g) - ({grid.y0!r})) / ({grid.dy!r})) AS INTEGER) AS iy,
    CAST(round((ST_X(g) - ({grid.x0!r})) / ({grid.dx!r})) AS INTEGER) AS ix
  FROM b
"""


def _write_manifest(s3: s3fs.S3FileSystem, record: dict) -> None:
    """Merge the record into FDP/buildings/manifest.json (keyed by id)."""
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


def load_grid() -> Grid:
    ds = xr.open_zarr(f'{store_uri()}/0', consolidated=False)
    print(f'  loading {NAME} level 0 {dict(ds.sizes)} into memory ...')
    return Grid(ds.load())


def cmd_build(args) -> None:
    s3 = s3fs.S3FileSystem()
    uri = output_uri()
    if not args.force and s3.exists(uri.removeprefix('s3://')):
        sys.exit(f'{uri} exists; use --force')
    if not s3.exists(f'{store_uri()}/0/zarr.json'.removeprefix('s3://')):
        sys.exit(f'source store missing: {store_uri()}')

    grid = load_grid()
    print(f'  lon/lat window {grid.lon0:.2f},{grid.lat0:.2f} -> {grid.lon1:.2f},{grid.lat1:.2f}')

    con = connect()

    # Pass 1 reads only `bbox`, so parquet column pruning keeps it off the
    # geometry column and it costs a fraction of the full scan.
    print('scanning buildings for occupied cells ...')
    occupied = con.sql(
        f'SELECT DISTINCT iy, ix FROM ({_indexed_sql(grid, "bbox")})'
    ).to_arrow_table()
    iy = occupied['iy'].to_numpy()
    ix = occupied['ix'].to_numpy()
    cells = grid.cells_at(iy, ix)
    print(f'  {len(iy):,} occupied cells, {cells.num_rows:,} with data')
    con.register('cells', cells)

    kv = {
        'store': store_uri(),
        'buildings_source': BUILDINGS_URI,
        'crs': CRS,
        'sampling': 'nearest 100 m cell at the footprint bbox centroid',
    }
    kv_sql = ', '.join(f"'{k}': '{v.replace(chr(39), chr(39) * 2)}'" for k, v in kv.items())
    cols = 'block_geoid AS GEOID, state_abbrev AS state, county_name AS county, bbox, geometry'
    query = f"""
COPY (
  SELECT s.GEOID, s.state, s.county, s.bbox, cells.{VARIABLE}, s.geometry
  FROM ({_indexed_sql(grid, cols)}) s
  JOIN cells ON cells.iy = s.iy AND cells.ix = s.ix
  ORDER BY s.iy, s.ix
) TO '{uri}' (FORMAT parquet, COMPRESSION zstd, KV_METADATA {{{kv_sql}}})
"""
    print(f'sampling buildings -> {uri}')
    con.sql(query)

    n, filled, lo, hi = con.sql(
        f'SELECT count(*), count({VARIABLE}), min({VARIABLE}), max({VARIABLE}) '
        f"FROM read_parquet('{uri}')"
    ).fetchone()
    print(f'wrote {n:,} buildings; {filled:,} with a value, range {lo:.4f}..{hi:.4f}')

    _write_manifest(
        s3,
        {
            'id': OUTPUT_ID,
            'path': uri,
            'columns': ['GEOID', 'state', 'county', 'bbox', VARIABLE, 'geometry'],
            'rows': n,
            'occupied_cells': cells.num_rows,
            **kv,
        },
    )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def cmd_verify(args) -> None:
    uri = output_uri()
    print(f'verifying {uri}')
    con = connect()
    rel = f"read_parquet('{uri}')"
    failures = 0

    kv = {
        (k.decode() if isinstance(k, bytes) else str(k)): (
            v.decode() if isinstance(v, bytes) else str(v)
        )
        for k, v in con.sql(f"SELECT key, value FROM parquet_kv_metadata('{uri}')").fetchall()
    }
    for field, want in (('store', store_uri()), ('crs', CRS), ('buildings_source', BUILDINGS_URI)):
        if kv.get(field) != want:
            print(f'STALE: metadata {field}={kv.get(field)!r}, this code says {want!r}')
            failures += 1

    n, filled, lo, hi = con.sql(
        f'SELECT count(*), count({VARIABLE}), min({VARIABLE}), max({VARIABLE}) FROM {rel}'
    ).fetchone()
    print(f'{n:,} rows, {filled:,} with a value, range {lo:.4f}..{hi:.4f}')
    if filled != n:
        print(f'  {n - filled:,} rows carry no value; the join should have dropped them')
        failures += 1
    if lo < 0 or hi > 1:
        print(f'  outside [0, 1]: {lo:.4f}..{hi:.4f}')
        failures += 1

    # bbox-filtered reads prune on row-group stats, so the output must be
    # spatially clustered; the build's grid-order sort provides it.
    meta = con.sql(
        'SELECT row_group_id, path_in_schema, stats_min_value::DOUBLE, stats_max_value::DOUBLE '
        f"FROM parquet_metadata('{uri}') WHERE path_in_schema IN ('bbox, xmin', 'bbox, ymin')"
    ).fetchall()
    env = {}
    for gid, col, mn, mx in meta:
        env.setdefault(gid, {})[col.removeprefix('bbox, ')] = mx - mn
    spans = [(g['xmin'], g['ymin']) for g in env.values() if 'xmin' in g and 'ymin' in g]
    med_w = float(np.median([w for w, _ in spans]))
    med_h = float(np.median([h for _, h in spans]))
    print(f'{len(spans)} row groups; median bbox-stat envelope {med_w:.2f} x {med_h:.2f} deg')
    if med_w > 5 and med_h > 5:
        print('  poorly clustered: bbox-filtered reads will scan most row groups')
        failures += 1

    sample = con.sql(
        f'SELECT bbox, {VARIABLE} FROM {rel} USING SAMPLE {args.n} ROWS'
    ).to_arrow_table()
    bbox = sample['bbox'].to_pylist()
    lon = np.array([(b['xmin'] + b['xmax']) / 2 for b in bbox])
    lat = np.array([(b['ymin'] + b['ymax']) / 2 for b in bbox])
    xs, ys = Transformer.from_crs('EPSG:4326', CRS, always_xy=True).transform(lon, lat)
    ds = xr.open_zarr(f'{store_uri()}/0', consolidated=False)
    want = (
        ds[VARIABLE]
        .sel(x=xr.DataArray(xs, dims='pt'), y=xr.DataArray(ys, dims='pt'), method='nearest')
        .values.astype('float64')
    )
    got = np.array(sample[VARIABLE].to_pylist(), dtype='float64')
    differ = int(
        (~(np.isclose(got, want, rtol=0, atol=0) | (np.isnan(got) & np.isnan(want)))).sum()
    )
    # nearest-neighbour ties at cell edges can legitimately disagree between
    # round() and xarray's sel; anything beyond a trace is a bug
    if differ > max(1, args.n // 1000):
        print(f'  {differ}/{len(got)} sampled rows disagree with xarray')
        failures += 1
    print(f'spot-checked {len(got)} rows against xarray nearest-sampling: {differ} differ')

    sys.exit(1 if failures else 0)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='sample the store onto buildings')
    b.add_argument('--force', action='store_true', help='overwrite an existing output')
    v = sub.add_parser('verify', help='check the written file')
    v.add_argument('--n', type=int, default=2000, help='rows to spot-check')
    args = p.parse_args()
    {'build': cmd_build, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
