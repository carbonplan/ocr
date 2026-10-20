# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1",
#   "mapbox-vector-tile",
#   "pmtiles",
#   "requests",
#   "s3fs>=2024.0.0",
# ]
# ///
"""Build the Google Groundsource flood event pmtiles for CONUS.

Source: Groundsource (Google Research, 2026), a CC BY 4.0 dataset of 2.6M
flood records Gemini extracted from news articles, published on Zenodo as
one GeoParquet file. Each record is a geocoded place polygon (a street,
neighborhood, town, or whole city) with a start and end date; there is no
article text, source, or damage.

Two artifacts share the same records and fields. `footprints` keeps each
polygon; `centroids` is a Point at each polygon's centroid. Records are kept
when their centroid falls inside the CONUS bounding box, so border areas of
Canada and Mexico are included.

Pipeline: fetch the parquet -> mirror raw to S3 -> CONUS subset with duckdb
-> GeoJSONL -> tippecanoe ->
s3://carbonplan-ocr/ocr-explore/groundsource_conus_footprints.pmtiles or
groundsource_conus_centroids.pmtiles, with a provenance record merged into
ocr-explore/manifest.json.

Requires tippecanoe on PATH and AWS credentials for carbonplan-ocr.

Usage:
  uv run groundsource_floods.py fetch
  uv run groundsource_floods.py build --artifact footprints|centroids [--force]
  uv run groundsource_floods.py verify --artifact footprints|centroids [--coverage]
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
RAW = f'{PREFIX}/Groundsource/raw'
MANIFEST_KEY = f'{PREFIX}/manifest.json'

ZENODO_RECORD = 'https://zenodo.org/api/records/18647054'
SOURCE_FILE = 'groundsource_2026.parquet'
CONUS_BBOX = (-125.0, 24.0, -66.0, 50.5)
ATTRIBUTION = 'Groundsource, Google Research (CC BY 4.0)'
LAYER = 'groundsource'
FIELDS = ['uuid', 'area_km2', 'start_date', 'end_date', 'year']
ARTIFACTS = {
    'footprints': {
        'id': 'groundsource_conus_footprints',
        'key': f'{PREFIX}/groundsource_conus_footprints.pmtiles',
        'name': 'Groundsource flood footprints',
        'geometry': 'geom',
        'maxzoom': 12,
    },
    'centroids': {
        'id': 'groundsource_conus_centroids',
        'key': f'{PREFIX}/groundsource_conus_centroids.pmtiles',
        'name': 'Groundsource flood centroids',
        'geometry': 'ST_Centroid(geom)',
        'maxzoom': 10,
    },
}
MINZOOM = 0

CACHE = Path.home() / '.cache' / 'ocr' / 'groundsource'
FETCH_INFO = CACHE / 'fetch_info.json'


def tippecanoe_args(artifact: dict) -> list[str]:
    maxzoom = str(artifact['maxzoom'])
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
        maxzoom,
        # keep every feature at every zoom unless a tile overflows
        '-r1',
        '--drop-densest-as-needed',
        '--no-tiny-polygon-reduction-at-maximum-zoom',
        '--force',
    ]


def artifact_uri(key: str) -> str:
    return f's3://{BUCKET}/{key}'


# ---------------------------------------------------------------------------
# Shared helpers (mirrored in input-data/vector/storm_events/storm_events_floods.py)
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
        'retrieved': datetime.date.today().isoformat(),
        'name': SOURCE_FILE,
        'size': entry['size'],
        'sha256': sha.hexdigest(),
        'raw_mirror': f's3://{raw_key}',
        'local': str(local),
    }
    FETCH_INFO.write_text(json.dumps(info, indent=2))
    print(f'{SOURCE_FILE}: {entry["size"] / 1e6:.0f} MB, mirrored to s3://{raw_key}')


def _write_geojsonl(source: Path, geometry: str, out: Path) -> dict:
    import duckdb

    con = duckdb.connect()
    con.execute('INSTALL spatial; LOAD spatial; SET enable_geoparquet_conversion = false')
    lo0, la0, lo1, la1 = CONUS_BBOX
    con.execute(
        f"""
        CREATE TABLE gs AS
        SELECT uuid, area_km2, start_date, end_date, ST_GeomFromWKB(geometry) AS geom
        FROM read_parquet('{source}')
        """
    )
    con.execute(
        f"""
        DELETE FROM gs WHERE NOT (
            ST_X(ST_Centroid(geom)) BETWEEN {lo0} AND {lo1}
            AND ST_Y(ST_Centroid(geom)) BETWEEN {la0} AND {la1}
        )
        """
    )
    count, y0, y1, *bounds = con.execute(
        """
        SELECT count(*), min(year(start_date::DATE)), max(year(start_date::DATE)),
            min(ST_XMin(geom)), min(ST_YMin(geom)), max(ST_XMax(geom)), max(ST_YMax(geom))
        FROM gs
        """
    ).fetchone()
    con.execute(
        f"""
        COPY (
            SELECT json_object(
                'type', 'Feature',
                'properties', json_object(
                    'uuid', uuid,
                    'area_km2', round(area_km2, 4),
                    'start_date', start_date,
                    'end_date', end_date,
                    'year', year(start_date::DATE)
                ),
                'geometry', ST_AsGeoJSON({geometry})::JSON
            )
            FROM gs
        ) TO '{out}' (FORMAT csv, HEADER false, QUOTE '', ESCAPE '')
        """
    )
    return {'count': count, 'years': [y0, y1], 'bounds': [round(v, 4) for v in bounds]}


def cmd_build(args) -> None:
    artifact = ARTIFACTS[args.artifact]
    key = artifact['key']
    s3 = s3fs.S3FileSystem()
    if s3.exists(f'{BUCKET}/{key}') and not args.force:
        sys.exit(f'{artifact_uri(key)} exists; use --force to overwrite')
    if not FETCH_INFO.exists():
        sys.exit('no fetched source; run `fetch` first')
    info = json.loads(FETCH_INFO.read_text())

    with tempfile.TemporaryDirectory(prefix='groundsource-') as tmp:
        geojsonl = Path(tmp) / 'groundsource.geojsonl'
        stats = _write_geojsonl(Path(info['local']), artifact['geometry'], geojsonl)
        print(f'{stats["count"]} records, {stats["years"]}, bounds={stats["bounds"]}')
        out = Path(tmp) / key.rsplit('/', 1)[-1]
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
            'subset': 'conus_bbox',
            'layer': LAYER,
            'fields': FIELDS,
            'feature_count': stats['count'],
            'years': stats['years'],
            'minzoom': MINZOOM,
            'maxzoom': artifact['maxzoom'],
            'bounds': stats['bounds'],
            'source': {
                k: info[k] for k in ('record', 'doi', 'license', 'retrieved', 'name', 'sha256')
            },
            'raw_mirror': info['raw_mirror'],
            'transform': (
                f'keep records whose polygon centroid lies in {list(CONUS_BBOX)}, write each '
                f'as {artifact["geometry"]} with its dates and area, and tile with tippecanoe'
            ),
            'tippecanoe': version,
            'tippecanoe_args': tippecanoe_args(artifact),
            'attribution': ATTRIBUTION,
            'display': {'promoteId': 'uuid', 'year_field': 'year'},
            'built': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'),
        },
    )


def cmd_verify(args) -> None:
    artifact = ARTIFACTS[args.artifact]
    key = artifact['key']
    s3 = s3fs.S3FileSystem()
    problems = []
    if not s3.exists(f'{BUCKET}/{key}'):
        sys.exit(f'{artifact_uri(key)} does not exist')
    header, meta = _pmtiles_metadata(_s3_get_bytes(s3, f'{BUCKET}/{key}'))
    layers = {vl['id']: vl for vl in meta.get('vector_layers', [])}
    if sorted(layers.get(LAYER, {}).get('fields', {})) != sorted(FIELDS):
        problems.append(f'layer {LAYER!r} fields differ from code')
    rec = _manifest_record(s3, artifact['id'])
    if rec is None:
        problems.append('no manifest record; rebuild to write provenance')
    else:
        if rec.get('tippecanoe_args') != tippecanoe_args(artifact):
            problems.append('manifest `tippecanoe_args` differs from this code')
        if args.coverage:
            local = CACHE / key.rsplit('/', 1)[-1]
            if not local.exists():
                s3.get(f'{BUCKET}/{key}', str(local))
            n = len(_maxzoom_ids(local, header['max_zoom'], 'uuid'))
            print(f'{n} of {rec["feature_count"]} records present at z{header["max_zoom"]}')
            if n != rec['feature_count']:
                problems.append(f'{rec["feature_count"] - n} records missing at max zoom')
    for p in problems:
        print(f'PROBLEM: {p}')
    print('ok' if not problems else f'{len(problems)} problem(s)')
    sys.exit(1 if problems else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fetch', help='download the parquet from Zenodo and mirror it to S3')
    b = sub.add_parser('build', help='build and upload one pmtiles + manifest record')
    b.add_argument('--force', action='store_true', help='overwrite the existing artifact')
    v = sub.add_parser('verify', help='check the served artifact against code and manifest')
    v.add_argument('--coverage', action='store_true', help='decode max-zoom tiles for drops')
    for parser in (b, v):
        parser.add_argument('--artifact', choices=sorted(ARTIFACTS), required=True)
    args = p.parse_args()
    {'fetch': cmd_fetch, 'build': cmd_build, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
