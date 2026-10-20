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
"""Build the United States Flood Database (USFD) point pmtiles for CONUS.

Source: USFD v1.1 (Li et al. 2021, ESSD, doi:10.5194/essd-13-3755-2021;
data doi:10.5281/zenodo.7545697, CC BY 4.0), one CSV merging seven flood
databases into a common format:

- FEDB: University of Connecticut flood events reconstructed from USGS
  streamflow and radar rainfall, 2002-2013, one point per gauged basin
- NOAA Storm report: NWS Storm Events, 1996-2020, one point per report
- GFM: Global Flood Monitor, flood tweets, 2014-2020
- mPing: crowdsourced reports from the NSSL mPing app, 2013-2017
- DFO, CyberFlood, EM-DAT, IFI: a few hundred events each or fewer

Rows with coordinates inside the CONUS bounding box are kept, each as a
Point. Row ids repeat in the source, so features carry their row number as
`id` and the source database's own id as `source_id`.

Pipeline: fetch the CSV -> mirror raw to S3 -> GeoJSONL -> tippecanoe ->
s3://carbonplan-ocr/ocr-explore/usfd_conus.pmtiles, with a provenance record
merged into ocr-explore/manifest.json.

Requires tippecanoe on PATH and AWS credentials for carbonplan-ocr.

Usage:
  uv run usfd_floods.py fetch
  uv run usfd_floods.py build [--force]
  uv run usfd_floods.py verify [--coverage]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
import s3fs

BUCKET = 'carbonplan-ocr'
PREFIX = 'ocr-explore'
RAW = f'{PREFIX}/USFD/raw'
MANIFEST_KEY = f'{PREFIX}/manifest.json'

ARTIFACT_ID = 'usfd_conus'
KEY = f'{PREFIX}/usfd_conus.pmtiles'
ZENODO_RECORD = 'https://zenodo.org/api/records/7545697'
SOURCE_FILE = 'USFD_v1.1.csv'
CONUS_BBOX = (-125.0, 24.0, -66.0, 50.5)
ATTRIBUTION = 'United States Flood Database v1.1, Li et al. 2021 (CC BY 4.0)'
LAYER = 'usfd'
FIELDS = [
    'id',
    'source',
    'source_id',
    'begin_date',
    'end_date',
    'year',
    'state',
    'location',
    'damage',
    'fatalities',
    'cause',
]
MINZOOM, MAXZOOM = 0, 10
TIPPECANOE_ARGS = [
    '-l',
    LAYER,
    '-n',
    'United States Flood Database',
    '-A',
    ATTRIBUTION,
    '-Z',
    str(MINZOOM),
    '-z',
    str(MAXZOOM),
    '-B',
    str(MAXZOOM),
    '--drop-densest-as-needed',
    '--force',
]

CACHE = Path.home() / '.cache' / 'ocr' / 'usfd'
FETCH_INFO = CACHE / 'fetch_info.json'


def artifact_uri(key: str) -> str:
    return f's3://{BUCKET}/{key}'


# ---------------------------------------------------------------------------
# Shared helpers (mirrored in input-data/vector/groundsource/groundsource_floods.py)
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


def _maxzoom_ids(local: Path, maxzoom: int, field: str) -> set:
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
                ids.add(feature['properties'][field])
    return ids


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


def _date(value) -> str | None:
    """'20020130013000.0', '199610211400', '20060407.0' -> 'YYYY-MM-DD'."""
    digits = str(value).split('.')[0]
    if len(digits) < 8 or not digits.isdigit():
        return None
    return f'{digits[:4]}-{digits[4:6]}-{digits[6:8]}'


def _text(value) -> str:
    return str(value).strip() if value == value and value is not None else ''


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _feature(row) -> dict:
    begin = _date(row.DATE_BEGIN)
    year = _number(row.year)
    props = {
        'id': int(row.Index),
        'source': row.SOURCE_DB,
        'source_id': _text(row.SOURCE_ID),
        'begin_date': begin,
        'end_date': _date(row.DATE_END),
        'year': int(year) if year else int(begin[:4]) if begin else None,
        'state': _text(row.STATE).title(),
        'location': _text(row.LOCATION).title(),
        'damage': _number(row.DAMAGE),
        'fatalities': _number(row.FATALITY),
        'cause': _text(row.CAUSE),
    }
    return {
        'type': 'Feature',
        'id': props['id'],
        'properties': {k: v for k, v in props.items() if v not in (None, '')},
        'geometry': {'type': 'Point', 'coordinates': [row.LON, row.LAT]},
    }


def _load(local: Path):
    import pandas as pd

    df = pd.read_csv(
        local,
        low_memory=False,
        dtype={'DATE_BEGIN': str, 'DATE_END': str, 'SOURCE_ID': str},
        usecols=[
            'DATE_BEGIN',
            'DATE_END',
            'LON',
            'LAT',
            'STATE',
            'LOCATION',
            'FATALITY',
            'DAMAGE',
            'CAUSE',
            'SOURCE_DB',
            'SOURCE_ID',
            'year',
        ],
    )
    total = len(df)
    lo0, la0, lo1, la1 = CONUS_BBOX
    df = df[df['LON'].between(lo0, lo1) & df['LAT'].between(la0, la1)]
    print(f'{len(df)} of {total} rows inside the CONUS bounding box')
    return df


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_fetch(_args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    record = requests.get(ZENODO_RECORD, timeout=60).json()
    entry = next(f for f in record['files'] if f['key'] == SOURCE_FILE)
    local = CACHE / SOURCE_FILE
    if not local.exists() or local.stat().st_size != entry['size']:
        with requests.get(entry['links']['self'], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(local, 'wb') as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(local, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            md5.update(chunk)
            sha.update(chunk)
    if f'md5:{md5.hexdigest()}' != entry['checksum']:
        sys.exit(f'{SOURCE_FILE}: checksum mismatch with Zenodo ({entry["checksum"]})')

    s3 = s3fs.S3FileSystem()
    raw_key = f'{BUCKET}/{RAW}/{SOURCE_FILE}'
    if not (s3.exists(raw_key) and s3.info(raw_key)['size'] == entry['size']):
        s3.put(str(local), raw_key)
    info = {
        'record': record['links']['self_html'],
        'doi': record.get('doi'),
        'license': record['metadata'].get('license', {}).get('id'),
        'version': record['metadata'].get('version'),
        'retrieved': datetime.date.today().isoformat(),
        'name': SOURCE_FILE,
        'size': entry['size'],
        'sha256': sha.hexdigest(),
        'raw_mirror': f's3://{raw_key}',
        'local': str(local),
    }
    FETCH_INFO.write_text(json.dumps(info, indent=2))
    print(f'{SOURCE_FILE}: {entry["size"] / 1e6:.0f} MB, mirrored to s3://{raw_key}')


def cmd_build(args) -> None:
    s3 = s3fs.S3FileSystem()
    if s3.exists(f'{BUCKET}/{KEY}') and not args.force:
        sys.exit(f'{artifact_uri(KEY)} exists; use --force to overwrite')
    if not FETCH_INFO.exists():
        sys.exit('no fetched source; run `fetch` first')
    info = json.loads(FETCH_INFO.read_text())

    df = _load(Path(info['local']))
    count = len(df)
    bounds = [
        round(float(v), 4)
        for v in (df['LON'].min(), df['LAT'].min(), df['LON'].max(), df['LAT'].max())
    ]
    sources = df['SOURCE_DB'].value_counts().to_dict()
    print(f'{count} points, bounds={bounds}, {sources}')

    with tempfile.TemporaryDirectory(prefix='usfd-') as tmp:
        geojsonl = Path(tmp) / 'usfd.geojsonl'
        out = Path(tmp) / KEY.rsplit('/', 1)[-1]
        with open(geojsonl, 'w') as fh:
            for row in df.itertuples():
                fh.write(json.dumps(_feature(row)) + '\n')
        version = _run_tippecanoe(geojsonl, out, TIPPECANOE_ARGS)
        print(f'{out.name}: {out.stat().st_size / 1e6:.1f} MB ({version})')
        shutil.copy(out, CACHE / out.name)
        s3.put(str(out), f'{BUCKET}/{KEY}')
        print(f'uploaded -> {artifact_uri(KEY)}')

    _write_manifest(
        s3,
        {
            'id': ARTIFACT_ID,
            'format': 'pmtiles',
            'url': f'https://{BUCKET}.s3.us-west-2.amazonaws.com/{KEY}',
            'subset': 'conus_bbox',
            'layer': LAYER,
            'fields': FIELDS,
            'feature_count': count,
            'sources': sources,
            'minzoom': MINZOOM,
            'maxzoom': MAXZOOM,
            'bounds': bounds,
            'source': {
                k: info[k]
                for k in ('record', 'doi', 'license', 'version', 'retrieved', 'name', 'sha256')
            },
            'raw_mirror': info['raw_mirror'],
            'transform': (
                f'keep rows with LON/LAT in {list(CONUS_BBOX)}, write each as a Point with '
                'its row number as id, dates parsed from DATE_BEGIN/DATE_END, and year from '
                'the year column or else DATE_BEGIN, and tile with tippecanoe'
            ),
            'tippecanoe': version,
            'tippecanoe_args': TIPPECANOE_ARGS,
            'attribution': ATTRIBUTION,
            'display': {'promoteId': 'id', 'category_field': 'source'},
            'built': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'),
        },
    )


def cmd_verify(args) -> None:
    s3 = s3fs.S3FileSystem()
    problems = []
    if not s3.exists(f'{BUCKET}/{KEY}'):
        sys.exit(f'{artifact_uri(KEY)} does not exist')
    header, meta = _pmtiles_metadata(_s3_get_bytes(s3, f'{BUCKET}/{KEY}'))
    layers = {vl['id']: vl for vl in meta.get('vector_layers', [])}
    if sorted(layers.get(LAYER, {}).get('fields', {})) != sorted(FIELDS):
        problems.append(f'layer {LAYER!r} fields differ from code')
    rec = _manifest_record(s3, ARTIFACT_ID)
    if rec is None:
        problems.append('no manifest record; rebuild to write provenance')
    else:
        if rec.get('tippecanoe_args') != TIPPECANOE_ARGS:
            problems.append('manifest `tippecanoe_args` differs from this code')
        if args.coverage:
            local = CACHE / KEY.rsplit('/', 1)[-1]
            if not local.exists():
                s3.get(f'{BUCKET}/{KEY}', str(local))
            n = len(_maxzoom_ids(local, header['max_zoom'], 'id'))
            print(f'{n} of {rec["feature_count"]} points present at z{header["max_zoom"]}')
            if n != rec['feature_count']:
                problems.append(f'{rec["feature_count"] - n} points missing at max zoom')
    for p in problems:
        print(f'PROBLEM: {p}')
    print('ok' if not problems else f'{len(problems)} problem(s)')
    sys.exit(1 if problems else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fetch', help='download the CSV from Zenodo and mirror it to S3')
    b = sub.add_parser('build', help='build and upload the pmtiles + manifest record')
    b.add_argument('--force', action='store_true', help='overwrite the existing artifact')
    v = sub.add_parser('verify', help='check the served artifact against code and manifest')
    v.add_argument('--coverage', action='store_true', help='decode max-zoom tiles for drops')
    args = p.parse_args()
    {'fetch': cmd_fetch, 'build': cmd_build, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
