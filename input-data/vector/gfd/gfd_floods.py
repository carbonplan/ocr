# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mapbox-vector-tile",
#   "numpy",
#   "pmtiles",
#   "rasterio",
#   "requests",
#   "s3fs>=2024.0.0",
#   "topozarr>=0.2",
#   "xarray",
#   "xproj",
#   "zarr>=3",
# ]
# ///
"""Build Global Flood Database flood footprints and counts for CONUS.

Source: Global Flood Database v1 (Tellman et al. 2021, Nature,
doi:10.1038/s41586-021-03695-w; CC BY-NC 4.0), 913 Dartmouth Flood
Observatory events from 2000 to 2018 mapped from MODIS at about 250 m. Cloud
to Street publishes each event as a GeoTIFF in the public gs://gfd_v3 bucket
with bands flooded, duration, clear_views, clear_perc, and jrc_perm_water,
and the event table (dates, country, cause, deaths, displaced) as
data/gfd_qcdatabase_2019_08_01.csv in cloudtostreet/MODIS_GlobalFloodDatabase.

A cell counts as flooded for an event when `flooded` is 1 and
`jrc_perm_water` is not. Events are kept when they flood a cell inside the
CONUS bounding box, so border areas of Canada and Mexico are included.

Artifacts, from one pass over the events:
- footprints: one Polygon per connected flooded patch, carrying its event's
  fields -> s3://carbonplan-ocr/ocr-explore/gfd_conus_footprints.pmtiles
- count / last_year: per-cell number of events and most recent event year
  (NaN where never flooded) on the native grid, as topozarr pyramids under
  s3://carbonplan-ocr/ocr-explore/GFD/

Requires tippecanoe on PATH and AWS credentials for carbonplan-ocr.

Usage:
  uv run gfd_floods.py fetch
  uv run gfd_floods.py build [--force]
  uv run gfd_floods.py verify [--coverage]
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
import s3fs

BUCKET = 'carbonplan-ocr'
PREFIX = 'ocr-explore'
RAW = f'{PREFIX}/GFD/raw'
MANIFEST_KEY = f'{PREFIX}/manifest.json'

SOURCE_BUCKET = 'gfd_v3'
SOURCE_LIST = f'https://storage.googleapis.com/storage/v1/b/{SOURCE_BUCKET}/o'
SOURCE_REPO = 'cloudtostreet/MODIS_GlobalFloodDatabase'
EVENT_TABLE = 'data/gfd_qcdatabase_2019_08_01.csv'
# large events are exported as several tiles, suffixed with row-col offsets
NAME_RE = re.compile(r'DFO_(\d+)_From_(\d{8})_to_(\d{8})(?:\d{10}-\d{10})?\.tif')
CONUS_BBOX = (-125.0, 24.0, -66.0, 50.5)
ATTRIBUTION = 'Global Flood Database, Tellman et al. 2021 (CC BY-NC 4.0)'

FOOTPRINTS = {
    'id': 'gfd_conus_footprints',
    'key': f'{PREFIX}/gfd_conus_footprints.pmtiles',
    'layer': 'gfd',
    'fields': [
        'dfo_id',
        'began',
        'ended',
        'year',
        'country',
        'cause',
        'severity',
        'dead',
        'displaced',
        'event_km2',
    ],
}
GRIDS = {
    'count': {'id': 'gfd_conus_flood_count', 'key': f'{PREFIX}/GFD/gfd_conus_flood_count'},
    'last_year': {
        'id': 'gfd_conus_last_flood_year',
        'key': f'{PREFIX}/GFD/gfd_conus_last_flood_year',
    },
}
MINZOOM, MAXZOOM = 0, 10
TIPPECANOE_ARGS = [
    '-l',
    FOOTPRINTS['layer'],
    '-n',
    'Global Flood Database flood footprints',
    '-A',
    ATTRIBUTION,
    '-Z',
    str(MINZOOM),
    '-z',
    str(MAXZOOM),
    '--drop-densest-as-needed',
    '--no-tiny-polygon-reduction-at-maximum-zoom',
    '--force',
]

CACHE = Path.home() / '.cache' / 'ocr' / 'gfd'
FETCH_INFO = CACHE / 'fetch_info.json'


def artifact_uri(key: str) -> str:
    return f's3://{BUCKET}/{key}'


def https_url(key: str) -> str:
    return f'https://{BUCKET}.s3.us-west-2.amazonaws.com/{key}'


# ---------------------------------------------------------------------------
# Shared helpers (mirrored in input-data/vector/usfd/usfd_floods.py)
# ---------------------------------------------------------------------------


def _run_tippecanoe(geojsonl: Path, out: Path, args: list[str]) -> str:
    proc = subprocess.run(['tippecanoe', '--version'], capture_output=True, text=True)
    version = (proc.stdout + proc.stderr).strip()
    subprocess.run(['tippecanoe', '-o', str(out), '-q', *args, str(geojsonl)], check=True)
    return version


def _pmtiles_metadata(get_bytes) -> tuple[dict, dict]:
    from pmtiles.reader import Reader

    reader = Reader(get_bytes)
    return reader.header(), reader.metadata()


def _s3_get_bytes(s3, path: str):
    def get_bytes(offset: int, length: int) -> bytes:
        with s3.open(path, 'rb') as fh:
            fh.seek(offset)
            return fh.read(length)

    return get_bytes


def _write_manifest(s3, record: dict) -> None:
    """Merge `record` into ocr-explore/manifest.json (keyed by id)."""
    key = f'{BUCKET}/{MANIFEST_KEY}'
    existing = {}
    if s3.exists(key):
        with s3.open(key, 'rb') as fh:
            existing = {r['id']: r for r in json.load(fh).get('artifacts', [])}
    existing[record['id']] = record
    doc = {'artifacts': sorted(existing.values(), key=lambda r: r['id'])}
    with s3.open(key, 'w') as fh:
        json.dump(doc, fh, indent=2)
    print(f'manifest: {len(doc["artifacts"])} artifacts -> s3://{key}')


def _manifest_record(s3, artifact_id: str) -> dict | None:
    key = f'{BUCKET}/{MANIFEST_KEY}'
    if not s3.exists(key):
        return None
    with s3.open(key, 'rb') as fh:
        return {r['id']: r for r in json.load(fh).get('artifacts', [])}.get(artifact_id)


def _maxzoom_ids(local: Path, maxzoom: int, layer: str, field: str) -> set:
    import gzip

    import mapbox_vector_tile
    from pmtiles.reader import MmapSource, all_tiles

    ids = set()
    with open(local, 'rb') as fh:
        for (z, _x, _y), data in all_tiles(MmapSource(fh)):
            if z != maxzoom:
                continue
            if data[:2] == b'\x1f\x8b':
                data = gzip.decompress(data)
            for feature in mapbox_vector_tile.decode(data)[layer]['features']:
                ids.add(feature['properties'][field])
    return ids


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


def _list_source() -> list[dict]:
    items, token = [], None
    while True:
        params = {'maxResults': 1000, 'fields': 'nextPageToken,items(name,size,md5Hash)'}
        if token:
            params['pageToken'] = token
        r = requests.get(SOURCE_LIST, params=params, timeout=60)
        r.raise_for_status()
        page = r.json()
        items += page.get('items', [])
        token = page.get('nextPageToken')
        if not token:
            return items


def _overlaps_conus(name: str) -> bool:
    import rasterio

    with rasterio.open(f'/vsicurl/https://storage.googleapis.com/{SOURCE_BUCKET}/{name}') as src:
        west, south, east, north = src.bounds
    lo0, la0, lo1, la1 = CONUS_BBOX
    return west < lo1 and east > lo0 and south < la1 and north > la0


def _fetch_tif(item: dict, s3) -> dict:
    local = CACHE / 'raw' / item['name']
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://storage.googleapis.com/{SOURCE_BUCKET}/{item["name"]}'
    for attempt in range(5):
        if local.exists() and local.stat().st_size == int(item['size']):
            break
        partial = local.with_suffix('.part')
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(partial, 'wb') as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            partial.rename(local)
        except requests.RequestException as e:
            print(f'{item["name"]}: attempt {attempt + 1} failed ({e})')
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(local, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            md5.update(chunk)
            sha.update(chunk)
    if base64.b64encode(md5.digest()).decode() != item['md5Hash']:
        sys.exit(f'{item["name"]}: md5 differs from gs://{SOURCE_BUCKET}')
    raw_key = f'{BUCKET}/{RAW}/{item["name"]}'
    if not (s3.exists(raw_key) and s3.info(raw_key)['size'] == int(item['size'])):
        s3.put(str(local), raw_key)
    print(f'{item["name"]}: {int(item["size"]) / 1e6:.0f} MB')
    return {'name': item['name'], 'sha256': sha.hexdigest(), 'local': str(local)}


def _events(info: dict) -> list[dict]:
    with open(info['event_table']['local'], newline='') as fh:
        table = {int(r['ID']): r for r in csv.DictReader(fh)}
    events = {}
    for f in info['files']:
        dfo_id, began, ended = NAME_RE.match(f['name']).groups()
        if int(dfo_id) in events:
            events[int(dfo_id)]['locals'].append(f['local'])
            continue
        row = table[int(dfo_id)]
        events[int(dfo_id)] = {
            'locals': [f['local']],
            'dfo_id': int(dfo_id),
            'began': f'{began[:4]}-{began[4:6]}-{began[6:]}',
            'ended': f'{ended[:4]}-{ended[4:6]}-{ended[6:]}',
            'year': int(began[:4]),
            'country': row['Country'],
            'cause': row['MainCause'].strip(),
            'severity': float(row['Severity']) if row['Severity'] else None,
            'dead': int(float(row['Dead'] or 0)),
            'displaced': int(float(row['Displaced'] or 0)),
        }
    return sorted(events.values(), key=lambda e: e['began'])


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _conus_grid(res: float):
    """Column/row bounds of the CONUS box on the global grid, whose cell edges
    fall on multiples of `res` from (0, 0)."""
    lo0, la0, lo1, la1 = CONUS_BBOX
    col0, col1 = math.floor(lo0 / res), math.ceil(lo1 / res)
    row0, row1 = math.floor(-la1 / res), math.ceil(-la0 / res)
    return col0, col1, row0, row1


def _polygons(mask, transform):
    from rasterio import features

    for geom, _ in features.shapes(mask.view('uint8'), mask=mask, transform=transform):
        yield {
            'type': 'Polygon',
            'coordinates': [
                [[round(x, 6), round(y, 6)] for x, y in ring] for ring in geom['coordinates']
            ],
        }


def _tile_mask(path: str, res: float, col0, col1, row0, row1):
    """The tile's flooded, non-permanent-water cells inside the CONUS grid, as
    (mask, transform, view into the CONUS grid), or None when it has none."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        if not math.isclose(src.res[0], res):
            sys.exit(f'{path}: resolution {src.res} differs')
        tcol = round(src.bounds.left / res)
        trow = round(-src.bounds.top / res)
        c0, c1 = max(col0, tcol), min(col1, tcol + src.width)
        r0, r1 = max(row0, trow), min(row1, trow + src.height)
        if c0 >= c1 or r0 >= r1:
            return None
        window = Window(c0 - tcol, r0 - trow, c1 - c0, r1 - r0)
        mask = (src.read(1, window=window) == 1) & (src.read(5, window=window) != 1)
        transform = src.window_transform(window)
    if not mask.any():
        return None
    return mask, transform, (slice(r0 - row0, r1 - row0), slice(c0 - col0, c1 - col0))


def _accumulate(events: list[dict], geojsonl: Path):
    import numpy as np
    import rasterio

    with rasterio.open(events[0]['locals'][0]) as src:
        res = src.res[0]
    col0, col1, row0, row1 = _conus_grid(res)
    count = np.zeros((row1 - row0, col1 - col0), dtype='uint8')
    last_year = np.zeros_like(count, dtype='uint16')
    kept = []
    with open(geojsonl, 'w') as out:
        for event in events:
            tiles = [_tile_mask(p, res, col0, col1, row0, row1) for p in event['locals']]
            tiles = [t for t in tiles if t is not None]
            if not tiles:
                continue
            event_km2 = 0.0
            for mask, transform, _ in tiles:
                lats = transform.f + (np.arange(mask.shape[0]) + 0.5) * transform.e
                cell_km2 = (res * 111.32) ** 2 * np.cos(np.radians(lats))
                event_km2 += float((mask.sum(axis=1) * cell_km2).sum())
            props = {k: v for k, v in event.items() if k != 'locals'}
            props = {k: v for k, v in props.items() if v is not None} | {
                'event_km2': round(event_km2, 1)
            }
            for mask, transform, view in tiles:
                for polygon in _polygons(mask, transform):
                    out.write(
                        json.dumps({'type': 'Feature', 'properties': props, 'geometry': polygon})
                        + '\n'
                    )
                count[view] += mask
                last_year[view] = np.where(
                    mask, np.maximum(last_year[view], event['year']), last_year[view]
                )
            kept.append(props)
            print(f'DFO {event["dfo_id"]} {event["began"]} {event["country"]}: {event_km2:.0f} km²')
    return count, last_year, kept, (res, col0, row0)


def _write_grid(values, grid, name: str, variable: str, tmp: Path, s3) -> dict:
    import numpy as np
    import xarray as xr
    import xproj  # noqa: F401
    from topozarr.coarsen import create_pyramid

    res, col0, row0 = grid
    data = np.where(values > 0, values, np.nan).astype('float32')
    x = (col0 + np.arange(data.shape[1]) + 0.5) * res
    y = -(row0 + np.arange(data.shape[0]) + 0.5) * res
    ds = xr.Dataset({variable: (('y', 'x'), data)}, coords={'x': x, 'y': y})
    ds = ds.proj.assign_crs(spatial_ref='EPSG:4326')
    pyramid = create_pyramid(ds, levels='auto', x_dim='x', y_dim='y', method='max')
    local = tmp / name
    pyramid.write(str(local), mode='w')
    key = GRIDS[name]['key']
    subprocess.run(
        ['aws', 's3', 'sync', str(local), artifact_uri(key), '--delete', '--only-show-errors'],
        check=True,
    )
    print(f'uploaded -> {artifact_uri(key)}')
    return {
        'shape': list(data.shape),
        'levels': int(pyramid.levels),
        'nonzero_cells': int(np.isfinite(data).sum()),
        'max': float(np.nanmax(data)),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_fetch(_args) -> None:
    os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    CACHE.mkdir(parents=True, exist_ok=True)
    s3 = s3fs.S3FileSystem()
    items = [i for i in _list_source() if NAME_RE.match(i['name'])]
    print(f'{len(items)} event GeoTIFFs in gs://{SOURCE_BUCKET}; reading bounds')
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(32) as pool:
        overlaps = list(pool.map(lambda i: _overlaps_conus(i['name']), items))
    items = [i for i, keep in zip(items, overlaps, strict=True) if keep]
    print(f'{len(items)} overlap the CONUS box')
    files = [_fetch_tif(item, s3) for item in items]

    sha = requests.get(f'https://api.github.com/repos/{SOURCE_REPO}/commits/HEAD', timeout=60)
    sha.raise_for_status()
    commit = sha.json()['sha']
    table_url = f'https://raw.githubusercontent.com/{SOURCE_REPO}/{commit}/{EVENT_TABLE}'
    table = requests.get(table_url, timeout=60)
    table.raise_for_status()
    table_local = CACHE / Path(EVENT_TABLE).name
    table_local.write_bytes(table.content)
    s3.pipe(f'{BUCKET}/{RAW}/{table_local.name}', table.content)

    info = {
        'bucket': f'gs://{SOURCE_BUCKET}',
        'retrieved': datetime.date.today().isoformat(),
        'files': files,
        'event_table': {
            'url': table_url,
            'sha256': hashlib.sha256(table.content).hexdigest(),
            'local': str(table_local),
        },
        'raw_mirror': f's3://{BUCKET}/{RAW}/',
    }
    FETCH_INFO.write_text(json.dumps(info, indent=2))
    print(f'{len(files)} files cached and mirrored to s3://{BUCKET}/{RAW}/')


def cmd_build(args) -> None:
    s3 = s3fs.S3FileSystem()
    if s3.exists(f'{BUCKET}/{FOOTPRINTS["key"]}') and not args.force:
        sys.exit(f'{artifact_uri(FOOTPRINTS["key"])} exists; use --force to overwrite')
    if not FETCH_INFO.exists():
        sys.exit('no fetched source; run `fetch` first')
    info = json.loads(FETCH_INFO.read_text())
    source = {
        'bucket': info['bucket'],
        'retrieved': info['retrieved'],
        'files': [{k: f[k] for k in ('name', 'sha256')} for f in info['files']],
        'event_table': {k: info['event_table'][k] for k in ('url', 'sha256')},
    }
    built = datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')

    with tempfile.TemporaryDirectory(prefix='gfd-') as tmp:
        tmp = Path(tmp)
        geojsonl = tmp / 'gfd.geojsonl'
        count, last_year, kept, grid = _accumulate(_events(info), geojsonl)
        years = [min(e['year'] for e in kept), max(e['year'] for e in kept)]
        print(f'{len(kept)} events flood the CONUS box, {years[0]}-{years[1]}')

        out = tmp / FOOTPRINTS['key'].rsplit('/', 1)[-1]
        version = _run_tippecanoe(geojsonl, out, TIPPECANOE_ARGS)
        print(f'{out.name}: {out.stat().st_size / 1e6:.1f} MB ({version})')
        shutil.copy(out, CACHE / out.name)
        s3.put(str(out), f'{BUCKET}/{FOOTPRINTS["key"]}')
        print(f'uploaded -> {artifact_uri(FOOTPRINTS["key"])}')
        _write_manifest(
            s3,
            {
                'id': FOOTPRINTS['id'],
                'format': 'pmtiles',
                'url': https_url(FOOTPRINTS['key']),
                'subset': 'conus_bbox',
                'layer': FOOTPRINTS['layer'],
                'fields': FOOTPRINTS['fields'],
                'event_count': len(kept),
                'event_ids': [e['dfo_id'] for e in kept],
                'years': years,
                'minzoom': MINZOOM,
                'maxzoom': MAXZOOM,
                'source': source,
                'raw_mirror': info['raw_mirror'],
                'transform': (
                    'per event, mark cells with flooded == 1 and jrc_perm_water != 1 inside '
                    f'{list(CONUS_BBOX)}, write each connected patch as a Polygon with the '
                    "event's fields and flooded area, and tile with tippecanoe"
                ),
                'tippecanoe': version,
                'tippecanoe_args': TIPPECANOE_ARGS,
                'attribution': ATTRIBUTION,
                'license': 'CC BY-NC 4.0',
                'display': {'promoteId': 'dfo_id', 'year_field': 'year'},
                'built': built,
            },
        )

        for name, values, variable, meaning in (
            ('count', count, 'count', 'number of events that flooded the cell'),
            ('last_year', last_year, 'year', 'year of the most recent event that flooded the cell'),
        ):
            stats = _write_grid(values, grid, name, variable, tmp, s3)
            _write_manifest(
                s3,
                {
                    'id': GRIDS[name]['id'],
                    'format': 'zarr-pyramid',
                    'url': https_url(GRIDS[name]['key']),
                    'subset': 'conus_bbox',
                    'variable': variable,
                    'meaning': f'{meaning}; NaN where no event flooded it',
                    'crs': 'EPSG:4326',
                    'resolution_deg': grid[0],
                    'coarsening': 'max',
                    **stats,
                    'event_count': len(kept),
                    'years': years,
                    'source': source,
                    'raw_mirror': info['raw_mirror'],
                    'attribution': ATTRIBUTION,
                    'license': 'CC BY-NC 4.0',
                    'built': built,
                },
            )


def cmd_verify(args) -> None:
    import zarr

    s3 = s3fs.S3FileSystem()
    problems = []
    rec = _manifest_record(s3, FOOTPRINTS['id'])
    if rec is None or not s3.exists(f'{BUCKET}/{FOOTPRINTS["key"]}'):
        sys.exit('footprints artifact or manifest record missing; run build')
    header, meta = _pmtiles_metadata(_s3_get_bytes(s3, f'{BUCKET}/{FOOTPRINTS["key"]}'))
    layers = {vl['id']: vl for vl in meta.get('vector_layers', [])}
    if sorted(layers.get(FOOTPRINTS['layer'], {}).get('fields', {})) != sorted(
        FOOTPRINTS['fields']
    ):
        problems.append('footprint fields differ from code')
    if rec.get('tippecanoe_args') != TIPPECANOE_ARGS:
        problems.append('manifest `tippecanoe_args` differs from this code')
    if args.coverage:
        local = CACHE / FOOTPRINTS['key'].rsplit('/', 1)[-1]
        if not local.exists():
            s3.get(f'{BUCKET}/{FOOTPRINTS["key"]}', str(local))
        ids = _maxzoom_ids(local, header['max_zoom'], FOOTPRINTS['layer'], 'dfo_id')
        missing = set(rec['event_ids']) - ids
        print(f'{len(ids)} of {rec["event_count"]} events present at z{header["max_zoom"]}')
        if missing:
            problems.append(f'events missing at max zoom: {sorted(missing)}')
    for name, grid in GRIDS.items():
        grec = _manifest_record(s3, grid['id'])
        if grec is None:
            problems.append(f'{name}: no manifest record')
            continue
        store = zarr.storage.FsspecStore.from_url(artifact_uri(grid['key']), read_only=True)
        level0 = zarr.open_group(store, mode='r', path='0')[grec['variable']]
        if list(level0.shape) != grec['shape']:
            problems.append(f'{name}: level 0 shape {level0.shape} != manifest {grec["shape"]}')
        print(f'{name}: level 0 {level0.shape}, manifest max {grec["max"]}')
    for p in problems:
        print(f'PROBLEM: {p}')
    print('ok' if not problems else f'{len(problems)} problem(s)')
    sys.exit(1 if problems else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fetch', help='download CONUS event GeoTIFFs and the event table, mirror to S3')
    b = sub.add_parser('build', help='build and upload footprints and grids + manifest records')
    b.add_argument('--force', action='store_true', help='overwrite existing artifacts')
    v = sub.add_parser('verify', help='check served artifacts against code and manifest')
    v.add_argument('--coverage', action='store_true', help='decode max-zoom tiles for drops')
    args = p.parse_args()
    {'fetch': cmd_fetch, 'build': cmd_build, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
