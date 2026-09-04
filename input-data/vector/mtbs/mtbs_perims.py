# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "geopandas",
#   "mapbox-vector-tile",
#   "pmtiles",
#   "pyogrio",
#   "requests",
#   "s3fs>=2024.0.0",
# ]
# ///
"""Build the MTBS fire burn perimeter pmtiles.

Source: MTBS burned area boundaries (USGS/USFS), the national shapefile of all
mapped fire perimeters (CONUS + AK + HI + PR, 1984-present). Each polygon
carries the incident name, ignition date, burned acres, and assessment type;
`year` is derived from the ignition date for map filtering.

Pipeline: fetch zip -> mirror raw to S3 -> reproject to EPSG:4326 ->
lowercase + select fields -> GeoJSONL -> tippecanoe ->
s3://carbonplan-ocr/ocr-explore/mtbs_perims.pmtiles, with a provenance record
merged into ocr-explore/manifest.json (keyed by id, read-modify-write, so
partial rebuilds never drop other artifacts' records).

Requires tippecanoe on PATH and AWS credentials for carbonplan-ocr.

Usage:
  uv run mtbs_perims.py fetch
  uv run mtbs_perims.py build [--force] [--subset conus]
  uv run mtbs_perims.py verify [--subset conus] [--coverage]
  uv run mtbs_perims.py list [--subset conus]
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
RAW = f'{PREFIX}/MTBS/raw'
MANIFEST_KEY = f'{PREFIX}/manifest.json'

ARTIFACTS = {
    'national': ('mtbs_perims', f'{PREFIX}/mtbs_perims.pmtiles'),
    'conus': ('mtbs_conus_perims', f'{PREFIX}/mtbs_conus_perims.pmtiles'),
}
CONUS_BBOX = (-125.0, 24.0, -66.0, 50.5)
SOURCE_URL = (
    'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/'
    'composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip'
)
ATTRIBUTION = 'MTBS (USGS/USFS)'
LAYER = 'mtbs_perims'
# Lowercased source columns; `year` is derived from ig_date.
FIELDS = ['event_id', 'incid_name', 'incid_type', 'ig_date', 'year', 'burnbndac', 'asmnt_type']
MINZOOM, MAXZOOM = 0, 11
TIPPECANOE_ARGS = [
    '-l',
    LAYER,
    '-n',
    'MTBS Burned Area Boundaries',
    '-A',
    ATTRIBUTION,
    '-Z',
    str(MINZOOM),
    '-z',
    str(MAXZOOM),
    '--drop-smallest-as-needed',
    '--force',
]

CACHE = Path.home() / '.cache' / 'ocr' / 'mtbs'
FETCH_INFO = CACHE / 'fetch_info.json'


def artifact_uri(key: str) -> str:
    return f's3://{BUCKET}/{key}'


# ---------------------------------------------------------------------------
# Shared helpers (mirrored in input-data/vector/ibtracs/ibtracs_tracks.py)
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
    import pandas as pd

    artifact_id, key = ARTIFACTS[args.subset]
    s3 = s3fs.S3FileSystem()
    if s3.exists(f'{BUCKET}/{key}') and not args.force:
        sys.exit(f'{artifact_uri(key)} exists; use --force to overwrite')
    if not FETCH_INFO.exists():
        sys.exit('no fetched source; run `fetch` first')
    info = json.loads(FETCH_INFO.read_text())

    print(f'reading {info["local"]}')
    gdf = gpd.read_file(info['local'])
    gdf.columns = [c.lower() for c in gdf.columns]
    gdf = gdf.to_crs('EPSG:4326')
    gdf['year'] = pd.to_datetime(gdf['ig_date'], errors='coerce').dt.year
    if not gdf['year'].isna().any():
        gdf['year'] = gdf['year'].astype(int)
    gdf = gdf[[*FIELDS, 'geometry']]
    if args.subset == 'conus':
        lo0, la0, lo1, la1 = CONUS_BBOX
        gdf = gdf.cx[lo0:lo1, la0:la1]
    count = len(gdf)
    bounds = [round(float(b), 4) for b in gdf.total_bounds]
    print(f'{count} perimeters, bounds={bounds}')

    with tempfile.TemporaryDirectory(prefix='mtbs-') as tmp:
        geojsonl = Path(tmp) / 'mtbs_perims.geojsonl'
        out = Path(tmp) / key.rsplit('/', 1)[-1]
        gdf.to_file(geojsonl, driver='GeoJSONSeq')
        version = _run_tippecanoe(geojsonl, out, TIPPECANOE_ARGS)
        print(f'{out.name}: {out.stat().st_size / 1e6:.1f} MB ({version})')
        shutil.copy(out, CACHE / out.name)
        s3.put(str(out), f'{BUCKET}/{key}')
        print(f'uploaded -> {artifact_uri(key)}')

    _write_manifest(
        s3,
        {
            'id': artifact_id,
            'format': 'pmtiles',
            'url': f'https://{BUCKET}.s3.us-west-2.amazonaws.com/{key}',
            'subset': args.subset,
            'layer': LAYER,
            'fields': FIELDS,
            'feature_count': count,
            'minzoom': MINZOOM,
            'maxzoom': MAXZOOM,
            'bounds': bounds,
            'source': {k: info[k] for k in ('url', 'retrieved', 'last_modified', 'sha256')},
            'raw_mirror': info['raw_mirror'],
            'transform': (
                'read the national MTBS burned-area-boundaries shapefile, reproject to '
                'EPSG:4326, lowercase and keep only the fields listed (deriving `year` '
                'from ig_date), and tile with tippecanoe'
            ),
            'tippecanoe': version,
            'tippecanoe_args': TIPPECANOE_ARGS,
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
    artifact_id, key = ARTIFACTS[args.subset]
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

    rec = _manifest_record(s3, artifact_id)
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
            print(f'{n} of {rec.get("feature_count")} perimeters present at z{header["max_zoom"]}')
            if n != rec.get('feature_count'):
                problems.append(f'{rec.get("feature_count") - n} perimeters missing at max zoom')
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


def cmd_list(args) -> None:
    artifact_id, key = ARTIFACTS[args.subset]
    s3 = s3fs.S3FileSystem()
    raw = s3.ls(f'{BUCKET}/{RAW}') if s3.exists(f'{BUCKET}/{RAW}') else []
    print(f'raw mirrors: {[r.rsplit("/", 1)[-1] for r in raw] or "none"}')
    if s3.exists(f'{BUCKET}/{key}'):
        print(f'artifact: {artifact_uri(key)} ({s3.info(f"{BUCKET}/{key}")["size"] / 1e6:.1f} MB)')
    else:
        print('artifact: missing')
    rec = _manifest_record(s3, artifact_id)
    print(f'manifest record: {json.dumps(rec, indent=2) if rec else "none"}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fetch', help='download the MTBS shapefile zip and mirror it to S3')
    b = sub.add_parser('build', help='build and upload the pmtiles + manifest record')
    b.add_argument('--force', action='store_true', help='overwrite the existing artifact')
    v = sub.add_parser('verify', help='check the served artifact against code and manifest')
    v.add_argument('--coverage', action='store_true', help='decode max-zoom tiles for drops')
    ls = sub.add_parser('list', help='show raw mirrors, artifact, and manifest record')
    for sp in (b, v, ls):
        sp.add_argument('--subset', choices=sorted(ARTIFACTS), default='national')
    args = p.parse_args()
    {'fetch': cmd_fetch, 'build': cmd_build, 'verify': cmd_verify, 'list': cmd_list}[args.cmd](args)


if __name__ == '__main__':
    main()
