# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "geopandas",
#   "pmtiles",
#   "pyogrio",
#   "requests",
#   "s3fs>=2024.0.0",
# ]
# ///
"""Build the IBTrACS since-1980 hurricane track pmtiles.

Source: NOAA NCEI IBTrACS v04r01, since-1980 subset, "lines" shapefile — one
LineString per 3/6-hourly track segment, each carrying its own Saffir-Simpson
category (USA_SSHS) so tracks recolor as storms intensify. The viewer
(ocr-explore datasets/ibtracs.tsx) reads layer `tracks` and promotes `SID` so
hover lights the whole storm.

Pipeline: fetch zip -> mirror raw to S3 -> select fields -> GeoJSONL ->
tippecanoe -> s3://carbonplan-ocr/ocr-explore/ibtracs_since1980.pmtiles,
with a provenance record merged into ocr-explore/manifest.json (keyed by id,
read-modify-write, so partial rebuilds never drop other artifacts' records).

Requires tippecanoe on PATH and AWS credentials for carbonplan-ocr.

Usage:
  uv run ibtracs_tracks.py fetch
  uv run ibtracs_tracks.py build [--force]
  uv run ibtracs_tracks.py verify
  uv run ibtracs_tracks.py list
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
import s3fs

BUCKET = 'carbonplan-ocr'
PREFIX = 'ocr-explore'
RAW = f'{PREFIX}/IBTrACS/raw'
KEY = f'{PREFIX}/ibtracs_since1980.pmtiles'
MANIFEST_KEY = f'{PREFIX}/manifest.json'

ARTIFACT_ID = 'ibtracs_since1980'
SOURCE_URL = (
    'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-'
    'ibtracs/v04r01/access/shapefile/IBTrACS.since1980.list.v04r01.lines.zip'
)
ATTRIBUTION = 'IBTrACS v04r01, NOAA NCEI'
LAYER = 'tracks'
# Everything the viewer's styling/popup touches; the rest of the ~60 source
# columns are dropped to keep tiles small.
FIELDS = ['SID', 'SEASON', 'BASIN', 'NAME', 'ISO_TIME', 'NATURE', 'USA_WIND', 'USA_SSHS']
MINZOOM, MAXZOOM = 0, 9
TIPPECANOE_ARGS = [
    '-l',
    LAYER,
    '-n',
    'IBTrACS since-1980 tropical cyclone tracks (v04r01)',
    '-A',
    ATTRIBUTION,
    '-Z',
    str(MINZOOM),
    '-z',
    str(MAXZOOM),
    '--no-tile-size-limit',
    '--no-feature-limit',
    '-P',
    '--force',
]

CACHE = Path.home() / '.cache' / 'ocr' / 'ibtracs'
FETCH_INFO = CACHE / 'fetch_info.json'


def artifact_uri() -> str:
    return f's3://{BUCKET}/{KEY}'


# ---------------------------------------------------------------------------
# Shared helpers (mirrored in input-data/vector/mtbs/mtbs_perims.py)
# ---------------------------------------------------------------------------


def _fetch_source(url: str, cache: Path, info_path: Path, raw_prefix: str) -> dict:
    """Download `url` into `cache`, mirror it to S3, and record fetch metadata."""
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / url.rsplit('/', 1)[-1]
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    sha = hashlib.sha256()
    size = 0
    with open(local, 'wb') as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
            sha.update(chunk)
            size += len(chunk)
    print(f'fetched {local.name}: {size / 1e6:.1f} MB')
    # Raw mirror keys are dated by the source's Last-Modified, so a refetch
    # after an upstream update never overwrites the input of a prior build.
    lm = r.headers.get('Last-Modified')
    stamp = (
        datetime.datetime.strptime(lm, '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y%m%d')
        if lm
        else datetime.date.today().strftime('%Y%m%d')
    )
    raw_key = f'{raw_prefix}/{local.stem}.src{stamp}{local.suffix}'
    s3 = s3fs.S3FileSystem()
    if s3.exists(f'{BUCKET}/{raw_key}') and s3.info(f'{BUCKET}/{raw_key}')['size'] == size:
        print(f'raw mirror exists: s3://{BUCKET}/{raw_key}')
    else:
        s3.put(str(local), f'{BUCKET}/{raw_key}')
        print(f'mirrored -> s3://{BUCKET}/{raw_key}')
    info = {
        'url': url,
        'retrieved': datetime.date.today().isoformat(),
        'last_modified': lm,
        'size': size,
        'sha256': sha.hexdigest(),
        'raw_mirror': f's3://{BUCKET}/{raw_key}',
        'local': str(local),
    }
    info_path.write_text(json.dumps(info, indent=2))
    return info


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
# Commands
# ---------------------------------------------------------------------------


def cmd_fetch(_args) -> None:
    _fetch_source(SOURCE_URL, CACHE, FETCH_INFO, RAW)


def cmd_build(args) -> None:
    import geopandas as gpd

    s3 = s3fs.S3FileSystem()
    if s3.exists(f'{BUCKET}/{KEY}') and not args.force:
        sys.exit(f'{artifact_uri()} exists; use --force to overwrite')
    if not FETCH_INFO.exists():
        sys.exit('no fetched source; run `fetch` first')
    info = json.loads(FETCH_INFO.read_text())

    print(f'reading {info["local"]}')
    gdf = gpd.read_file(info['local'], columns=FIELDS)
    gdf = gdf[[*FIELDS, 'geometry']]
    count = len(gdf)
    bounds = [round(float(b), 4) for b in gdf.total_bounds]
    print(f'{count} track segments, bounds={bounds}')

    with tempfile.TemporaryDirectory(prefix='ibtracs-') as tmp:
        geojsonl = Path(tmp) / 'tracks.geojsonl'
        out = Path(tmp) / 'ibtracs_since1980.pmtiles'
        gdf.to_file(geojsonl, driver='GeoJSONSeq')
        version = _run_tippecanoe(geojsonl, out, TIPPECANOE_ARGS)
        print(f'{out.name}: {out.stat().st_size / 1e6:.1f} MB ({version})')
        s3.put(str(out), f'{BUCKET}/{KEY}')
        print(f'uploaded -> {artifact_uri()}')

    _write_manifest(
        s3,
        {
            'id': ARTIFACT_ID,
            'format': 'pmtiles',
            'url': f'https://{BUCKET}.s3.us-west-2.amazonaws.com/{KEY}',
            'layer': LAYER,
            'fields': FIELDS,
            'feature_count': count,
            'minzoom': MINZOOM,
            'maxzoom': MAXZOOM,
            'bounds': bounds,
            'source': {k: info[k] for k in ('url', 'retrieved', 'last_modified', 'sha256')},
            'raw_mirror': info['raw_mirror'],
            'transform': (
                'read the NCEI since-1980 lines shapefile (one LineString per 3/6-hourly '
                'track segment, already EPSG:4326), keep only the fields listed, and tile '
                'with tippecanoe'
            ),
            'tippecanoe': version,
            'tippecanoe_args': TIPPECANOE_ARGS,
            'attribution': ATTRIBUTION,
            'display': {'promoteId': 'SID', 'category_field': 'USA_SSHS'},
            'built': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'),
        },
    )


def cmd_verify(_args) -> None:
    s3 = s3fs.S3FileSystem()
    problems = []
    if not s3.exists(f'{BUCKET}/{KEY}'):
        sys.exit(f'{artifact_uri()} does not exist')
    header, meta = _pmtiles_metadata(_s3_get_bytes(s3, f'{BUCKET}/{KEY}'))
    layers = {vl['id']: vl for vl in meta.get('vector_layers', [])}
    if LAYER not in layers:
        problems.append(f'layer {LAYER!r} missing (has {sorted(layers)})')
    else:
        got = sorted(layers[LAYER].get('fields', {}))
        if got != sorted(FIELDS):
            problems.append(f'fields differ from code: {got}')
    if (header['min_zoom'], header['max_zoom']) != (MINZOOM, MAXZOOM):
        problems.append(f'zooms {header["min_zoom"]}-{header["max_zoom"]}')

    rec = _manifest_record(s3, ARTIFACT_ID)
    if rec is None:
        problems.append('no manifest record; rebuild to write provenance')
    else:
        tilestats = {layer['layer']: layer for layer in meta.get('tilestats', {}).get('layers', [])}
        count = tilestats.get(LAYER, {}).get('count')
        if count != rec.get('feature_count'):
            problems.append(f'tilestats count {count} != manifest {rec.get("feature_count")}')
        for field, want in (('fields', FIELDS), ('tippecanoe_args', TIPPECANOE_ARGS)):
            got = rec.get(field)
            if got is None:
                problems.append(f'manifest record predates `{field}`')
            elif got != want:
                problems.append(f'manifest `{field}` differs from this code ({got!r})')
    for p in problems:
        print(f'PROBLEM: {p}')
    print('ok' if not problems else f'{len(problems)} problem(s)')
    sys.exit(1 if problems else 0)


def cmd_list(_args) -> None:
    s3 = s3fs.S3FileSystem()
    raw = s3.ls(f'{BUCKET}/{RAW}') if s3.exists(f'{BUCKET}/{RAW}') else []
    print(f'raw mirrors: {[r.rsplit("/", 1)[-1] for r in raw] or "none"}')
    if s3.exists(f'{BUCKET}/{KEY}'):
        print(f'artifact: {artifact_uri()} ({s3.info(f"{BUCKET}/{KEY}")["size"] / 1e6:.1f} MB)')
    else:
        print('artifact: missing')
    rec = _manifest_record(s3, ARTIFACT_ID)
    print(f'manifest record: {json.dumps(rec, indent=2) if rec else "none"}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fetch', help='download the NCEI shapefile zip and mirror it to S3')
    b = sub.add_parser('build', help='build and upload the pmtiles + manifest record')
    b.add_argument('--force', action='store_true', help='overwrite the existing artifact')
    sub.add_parser('verify', help='check the served artifact against code and manifest')
    sub.add_parser('list', help='show raw mirrors, artifact, and manifest record')
    args = p.parse_args()
    {'fetch': cmd_fetch, 'build': cmd_build, 'verify': cmd_verify, 'list': cmd_list}[args.cmd](args)


if __name__ == '__main__':
    main()
