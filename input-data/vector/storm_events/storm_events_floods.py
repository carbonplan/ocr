# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mapbox-vector-tile",
#   "pandas",
#   "pmtiles",
#   "requests",
#   "s3fs>=2024.0.0",
# ]
# ///
"""Build the NOAA Storm Events flood report pmtiles.

Source: NOAA NCEI Storm Events Database bulk "details" CSVs, one gzipped file
per year. Flood and Flash Flood events are recorded from 1996; each carries a
begin and end location that NWS Storm Data publishes as the two corners
bounding the flooded area (NWSI 10-1605, section 2.4). Reports without both
locations are dropped, which removes most of 1996-2005.

Two artifacts share the same reports and fields. In `boxes`, each report
becomes a Polygon (its bounding box), a LineString between its corners when
one side is under 100 m, or a Point at its center when both are. In
`points`, each report is a Point at its begin location, the cell Collins et
al. (2022) sampled as a damage presence.

Pipeline: fetch every year's latest CSV -> mirror raw to S3 -> filter to
CONUS Flood / Flash Flood with locations -> GeoJSONL -> tippecanoe ->
s3://carbonplan-ocr/ocr-explore/storm_events_conus_floods.pmtiles (boxes) or
storm_events_conus_flood_points.pmtiles (points), with a provenance record merged into ocr-explore/manifest.json (keyed by id,
read-modify-write, so partial rebuilds never drop other artifacts' records).

Requires tippecanoe on PATH and AWS credentials for carbonplan-ocr.

Usage:
  uv run storm_events_floods.py fetch
  uv run storm_events_floods.py build [--artifact boxes|points] [--force]
  uv run storm_events_floods.py verify [--artifact boxes|points] [--coverage]
  uv run storm_events_floods.py list [--artifact boxes|points]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
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
RAW = f'{PREFIX}/StormEvents/raw'
MANIFEST_KEY = f'{PREFIX}/manifest.json'

ARTIFACTS = {
    'boxes': {
        'id': 'storm_events_conus_floods',
        'key': f'{PREFIX}/storm_events_conus_floods.pmtiles',
        'name': 'NOAA Storm Events flood reports',
        'transform': (
            'concatenate the yearly details CSVs, keep Flood and Flash Flood events with '
            'begin and end coordinates inside CONUS, write each as its begin/end bounding '
            'box (a LineString between the corners when one side is under 100 m, a Point '
            'at its center when both are) with damage summed over property and crops '
            '(omitted when neither was entered) and deaths/injuries summed over direct '
            'and indirect, and tile with tippecanoe'
        ),
    },
    'points': {
        'id': 'storm_events_conus_flood_points',
        'key': f'{PREFIX}/storm_events_conus_flood_points.pmtiles',
        'name': 'NOAA Storm Events flood report begin points',
        'transform': (
            'concatenate the yearly details CSVs, keep Flood and Flash Flood events with '
            'begin and end coordinates inside CONUS, write each as a Point at its begin '
            'location with damage summed over property and crops (omitted when neither '
            'was entered) and deaths/injuries summed over direct and indirect, and tile '
            'with tippecanoe'
        ),
    },
}
SOURCE_INDEX = 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'
DETAILS_RE = re.compile(r'StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz')
FIRST_YEAR = 1996
EVENT_TYPES = {'Flood', 'Flash Flood'}
CONUS_BBOX = (-125.0, 24.0, -66.0, 50.5)
# A box side shorter than this is coordinate jitter, not extent
POINT_BELOW_M = 100
NON_CONUS_STATES = {
    'ALASKA',
    'HAWAII',
    'PUERTO RICO',
    'VIRGIN ISLANDS',
    'GUAM',
    'AMERICAN SAMOA',
}
ATTRIBUTION = 'NOAA NCEI Storm Events Database'
LAYER = 'floods'
FIELDS = [
    'event_id',
    'episode_id',
    'type',
    'begin_date',
    'end_date',
    'year',
    'state',
    'county',
    'location',
    'cause',
    'damage',
    'deaths',
    'injuries',
]
MINZOOM, MAXZOOM = 0, 10


def tippecanoe_args(artifact: dict) -> list[str]:
    return [
        '-l',
        LAYER,
        '-n',
        artifact['name'],
        '-A',
        ATTRIBUTION,
        '-Z',
        str(MINZOOM),
        '-z',
        str(MAXZOOM),
        '-B',
        str(MAXZOOM),
        '--drop-densest-as-needed',
        '--no-tiny-polygon-reduction-at-maximum-zoom',
        '--force',
    ]


CACHE = Path.home() / '.cache' / 'ocr' / 'storm-events'
FETCH_INFO = CACHE / 'fetch_info.json'


def artifact_uri(key: str) -> str:
    return f's3://{BUCKET}/{key}'


# ---------------------------------------------------------------------------
# Shared helpers (mirrored in input-data/vector/mtbs/mtbs_perims.py)
# ---------------------------------------------------------------------------


def _run_tippecanoe(geojsonl: Path, out: Path, args: list[str]) -> str:
    proc = subprocess.run(['tippecanoe', '--version'], capture_output=True, text=True)
    version = (proc.stdout + proc.stderr).strip()
    subprocess.run(['tippecanoe', '-o', str(out), *args, str(geojsonl)], check=True)
    return version


def _pmtiles_metadata(get_bytes) -> tuple[dict, dict]:
    """(header, metadata) of a pmtiles archive via a (offset, length) reader."""
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


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


def _latest_details_files() -> dict[int, str]:
    """Most recent details CSV per year, by NCEI's creation stamp."""
    r = requests.get(SOURCE_INDEX, timeout=60)
    r.raise_for_status()
    latest: dict[int, tuple[str, str]] = {}
    for name, year, stamp in {(m.group(0), *m.groups()) for m in DETAILS_RE.finditer(r.text)}:
        year = int(year)
        if year >= FIRST_YEAR and (year not in latest or stamp > latest[year][1]):
            latest[year] = (name, stamp)
    return {year: name for year, (name, _) in sorted(latest.items())}


def _fetch_file(name: str, s3) -> dict:
    """Download one CSV into the cache and mirror it to S3. The creation stamp
    in each file name makes raw mirror keys immutable across refetches."""
    local = CACHE / 'raw' / name
    local.parent.mkdir(parents=True, exist_ok=True)
    url = SOURCE_INDEX + name
    sha = hashlib.sha256()
    if not local.exists():
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(local, 'wb') as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    with open(local, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            sha.update(chunk)
    size = local.stat().st_size
    raw_key = f'{RAW}/{name}'
    if s3.exists(f'{BUCKET}/{raw_key}') and s3.info(f'{BUCKET}/{raw_key}')['size'] == size:
        status = 'mirror exists'
    else:
        s3.put(str(local), f'{BUCKET}/{raw_key}')
        status = 'mirrored'
    print(f'{name}: {size / 1e6:.1f} MB, {status}')
    return {
        'name': name,
        'url': url,
        'size': size,
        'sha256': sha.hexdigest(),
        'raw_mirror': f's3://{BUCKET}/{raw_key}',
        'local': str(local),
    }


def _parse_damage(value) -> float | None:
    """Storm Data damage strings like '10.00K', '2.5M', '0' -> dollars; None
    when no estimate was entered."""
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip().upper()
    scale = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'H': 1e2}.get(value[-1])
    number = value[:-1] if scale else value
    try:
        return float(number or 0) * (scale or 1)
    except ValueError:
        return None


def _date(yearmonth, day) -> str:
    ym = str(int(yearmonth))
    return f'{ym[:4]}-{ym[4:]}-{int(day):02d}'


def _geometry(row) -> dict:
    west, east = sorted((row.BEGIN_LON, row.END_LON))
    south, north = sorted((row.BEGIN_LAT, row.END_LAT))
    height = (north - south) * 111_320
    width = (east - west) * 111_320 * math.cos(math.radians((south + north) / 2))
    if max(height, width) < POINT_BELOW_M:
        return {'type': 'Point', 'coordinates': [(west + east) / 2, (south + north) / 2]}
    if min(height, width) < POINT_BELOW_M:
        coords = [[row.BEGIN_LON, row.BEGIN_LAT], [row.END_LON, row.END_LAT]]
        return {'type': 'LineString', 'coordinates': coords}
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {'type': 'Polygon', 'coordinates': [ring]}


def _load_reports(files: list[dict]):
    import pandas as pd

    frames = []
    for f in files:
        df = pd.read_csv(
            f['local'], low_memory=False, dtype={'DAMAGE_PROPERTY': str, 'DAMAGE_CROPS': str}
        )
        frames.append(df[df['EVENT_TYPE'].isin(EVENT_TYPES)])
    df = pd.concat(frames, ignore_index=True)
    total = len(df)
    df = df.dropna(subset=['BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON'])
    df = df[~df['STATE'].isin(NON_CONUS_STATES)]
    lo0, la0, lo1, la1 = CONUS_BBOX
    in_bbox = (
        df['BEGIN_LON'].between(lo0, lo1)
        & df['END_LON'].between(lo0, lo1)
        & df['BEGIN_LAT'].between(la0, la1)
        & df['END_LAT'].between(la0, la1)
    )
    df = df[in_bbox]
    if df['EVENT_ID'].duplicated().any():
        sys.exit('duplicate EVENT_IDs across yearly files')
    print(f'{len(df)} of {total} Flood / Flash Flood reports kept')
    return df


def _begin_point(row) -> dict:
    return {'type': 'Point', 'coordinates': [row.BEGIN_LON, row.BEGIN_LAT]}


GEOMETRY = {'boxes': _geometry, 'points': _begin_point}


def _feature(row, geometry) -> dict:
    amounts = [_parse_damage(v) for v in (row.DAMAGE_PROPERTY, row.DAMAGE_CROPS)]
    known = [a for a in amounts if a is not None]
    props = {
        'event_id': int(row.EVENT_ID),
        'episode_id': int(row.EPISODE_ID) if row.EPISODE_ID == row.EPISODE_ID else None,
        'type': row.EVENT_TYPE,
        'begin_date': _date(row.BEGIN_YEARMONTH, row.BEGIN_DAY),
        'end_date': _date(row.END_YEARMONTH, row.END_DAY),
        'year': int(row.YEAR),
        'state': str(row.STATE).title(),
        'county': str(row.CZ_NAME).title() if row.CZ_NAME == row.CZ_NAME else '',
        'location': str(row.BEGIN_LOCATION).title()
        if row.BEGIN_LOCATION == row.BEGIN_LOCATION
        else '',
        'cause': row.FLOOD_CAUSE if row.FLOOD_CAUSE == row.FLOOD_CAUSE else '',
        'damage': round(sum(known)) if known else None,
        'deaths': int(row.DEATHS_DIRECT + row.DEATHS_INDIRECT),
        'injuries': int(row.INJURIES_DIRECT + row.INJURIES_INDIRECT),
    }
    return {
        'type': 'Feature',
        'id': props['event_id'],
        'properties': props,
        'geometry': geometry(row),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_fetch(_args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    s3 = s3fs.S3FileSystem()
    files = [_fetch_file(name, s3) for name in _latest_details_files().values()]
    info = {
        'index': SOURCE_INDEX,
        'retrieved': datetime.date.today().isoformat(),
        'files': files,
    }
    FETCH_INFO.write_text(json.dumps(info, indent=2))
    print(f'{len(files)} files, {sum(f["size"] for f in files) / 1e6:.0f} MB')


def cmd_build(args) -> None:
    artifact = ARTIFACTS[args.artifact]
    key = artifact['key']
    s3 = s3fs.S3FileSystem()
    if s3.exists(f'{BUCKET}/{key}') and not args.force:
        sys.exit(f'{artifact_uri(key)} exists; use --force to overwrite')
    if not FETCH_INFO.exists():
        sys.exit('no fetched source; run `fetch` first')
    info = json.loads(FETCH_INFO.read_text())

    df = _load_reports(info['files'])
    count = len(df)
    bounds = [
        round(float(v), 4)
        for v in (
            min(df['BEGIN_LON'].min(), df['END_LON'].min()),
            min(df['BEGIN_LAT'].min(), df['END_LAT'].min()),
            max(df['BEGIN_LON'].max(), df['END_LON'].max()),
            max(df['BEGIN_LAT'].max(), df['END_LAT'].max()),
        )
    ]
    years = [int(df['YEAR'].min()), int(df['YEAR'].max())]
    print(f'{count} reports, {years[0]}-{years[1]}, bounds={bounds}')

    with tempfile.TemporaryDirectory(prefix='storm-events-') as tmp:
        geojsonl = Path(tmp) / 'floods.geojsonl'
        out = Path(tmp) / key.rsplit('/', 1)[-1]
        geometry = GEOMETRY[args.artifact]
        with open(geojsonl, 'w') as fh:
            for row in df.itertuples(index=False):
                fh.write(json.dumps(_feature(row, geometry)) + '\n')
        version = _run_tippecanoe(geojsonl, out, tippecanoe_args(artifact))
        print(f'{out.name}: {out.stat().st_size / 1e6:.1f} MB ({version})')
        shutil.copy(out, CACHE / out.name)
        s3.put(str(out), f'{BUCKET}/{key}')
        print(f'uploaded -> {artifact_uri(key)}')

    _write_manifest(
        s3,
        {
            'id': artifact['id'],
            'format': 'pmtiles',
            'url': f'https://{BUCKET}.s3.us-west-2.amazonaws.com/{key}',
            'subset': 'conus',
            'layer': LAYER,
            'fields': FIELDS,
            'feature_count': count,
            'years': years,
            'minzoom': MINZOOM,
            'maxzoom': MAXZOOM,
            'bounds': bounds,
            'source': {
                'index': info['index'],
                'retrieved': info['retrieved'],
                'files': [{k: f[k] for k in ('name', 'sha256')} for f in info['files']],
            },
            'raw_mirror': f's3://{BUCKET}/{RAW}/',
            'transform': artifact['transform'],
            'tippecanoe': version,
            'tippecanoe_args': tippecanoe_args(artifact),
            'attribution': ATTRIBUTION,
            'display': {'promoteId': 'event_id', 'year_field': 'year'},
            'built': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'),
        },
    )


def _maxzoom_ids(local: Path, maxzoom: int) -> set:
    """Every event_id present in some max-zoom tile, where nothing should
    have been dropped for size."""
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
            for feature in mapbox_vector_tile.decode(data)[LAYER]['features']:
                ids.add(feature['properties']['event_id'])
    return ids


def cmd_verify(args) -> None:
    artifact = ARTIFACTS[args.artifact]
    key = artifact['key']
    s3 = s3fs.S3FileSystem()
    problems = []
    if not s3.exists(f'{BUCKET}/{key}'):
        sys.exit(f'{artifact_uri(key)} does not exist')
    header, meta = _pmtiles_metadata(_s3_get_bytes(s3, f'{BUCKET}/{key}'))
    layers = {vl['id']: vl for vl in meta.get('vector_layers', [])}
    if LAYER not in layers:
        problems.append(f'layer {LAYER!r} missing (has {sorted(layers)})')
    else:
        got = sorted(layers[LAYER].get('fields', {}))
        if got != sorted(FIELDS):
            problems.append(f'fields differ from code: {got}')
    if (header['min_zoom'], header['max_zoom']) != (MINZOOM, MAXZOOM):
        problems.append(f'zooms {header["min_zoom"]}-{header["max_zoom"]}')

    rec = _manifest_record(s3, artifact['id'])
    if rec is None:
        problems.append('no manifest record; rebuild to write provenance')
    else:
        tilestats = {layer['layer']: layer for layer in meta.get('tilestats', {}).get('layers', [])}
        count = tilestats.get(LAYER, {}).get('count')
        if count != rec.get('feature_count'):
            problems.append(f'tilestats count {count} != manifest {rec.get("feature_count")}')
        if args.coverage:
            local = CACHE / key.rsplit('/', 1)[-1]
            if not local.exists():
                s3.get(f'{BUCKET}/{key}', str(local))
            n = len(_maxzoom_ids(local, header['max_zoom']))
            print(f'{n} of {rec.get("feature_count")} reports present at z{header["max_zoom"]}')
            if n != rec.get('feature_count'):
                problems.append(f'{rec.get("feature_count") - n} reports missing at max zoom')
        for field, want in (('fields', FIELDS), ('tippecanoe_args', tippecanoe_args(artifact))):
            got = rec.get(field)
            if got is None:
                problems.append(f'manifest record predates `{field}`')
            elif got != want:
                problems.append(f'manifest `{field}` differs from this code ({got!r})')
    for p in problems:
        print(f'PROBLEM: {p}')
    print('ok' if not problems else f'{len(problems)} problem(s)')
    sys.exit(1 if problems else 0)


def cmd_list(args) -> None:
    artifact = ARTIFACTS[args.artifact]
    key = artifact['key']
    s3 = s3fs.S3FileSystem()
    raw = s3.ls(f'{BUCKET}/{RAW}') if s3.exists(f'{BUCKET}/{RAW}') else []
    print(f'raw mirrors: {len(raw)} files')
    if s3.exists(f'{BUCKET}/{key}'):
        print(f'artifact: {artifact_uri(key)} ({s3.info(f"{BUCKET}/{key}")["size"] / 1e6:.1f} MB)')
    else:
        print('artifact: missing')
    rec = _manifest_record(s3, artifact['id'])
    print(f'manifest record: {json.dumps(rec, indent=2) if rec else "none"}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fetch', help="download each year's latest details CSV and mirror it to S3")
    b = sub.add_parser('build', help='build and upload the pmtiles + manifest record')
    b.add_argument('--force', action='store_true', help='overwrite the existing artifact')
    v = sub.add_parser('verify', help='check the served artifact against code and manifest')
    v.add_argument('--coverage', action='store_true', help='decode max-zoom tiles for drops')
    lst = sub.add_parser('list', help='show raw mirrors, artifact, and manifest record')
    for parser in (b, v, lst):
        parser.add_argument('--artifact', choices=sorted(ARTIFACTS), default='boxes')
    args = p.parse_args()
    {'fetch': cmd_fetch, 'build': cmd_build, 'verify': cmd_verify, 'list': cmd_list}[args.cmd](args)


if __name__ == '__main__':
    main()
