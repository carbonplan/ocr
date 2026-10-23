# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1",
#   "numpy",
#   "pyproj",
#   "rasterio",
#   "requests",
#   "s3fs>=2024.0.0",
# ]
# ///
"""Stage the Collins et al. (2022) model inputs in S3 for retraining.

Copies the eleven predictor rasters and the released presence/absence points
from the USGS data release (doi:10.5066/P954TTQN) into `FDP/inputs/`, and
rasterizes the HUC-2 watersheds onto the predictor grid.

The predictors are rewritten as tiled, deflate-compressed COGs so a training
task can window-read one watershed straight from S3. Values, dtype and nodata
are unchanged.

The HUC raster assigns every cell to exactly one watershed. Training masks each
watershed's predictors with it and the national mosaic takes each cell from its
watershed's model, so the 18 surfaces tile without gaps or double coverage.
Polygons come from the USGS WBD service generalized to 50 m, half a cell;
where generalized neighbours overlap, the higher HUC code wins.

Every step is skipped when its output is already recorded in the manifest, so
`stage` resumes after an interruption. Expect ~18 GB of ScienceBase downloads;
run it in us-west-2 next to the bucket (see the README).

Usage:
  python fdp_inputs.py stage
  python fdp_inputs.py verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import rasterio
import rasterio.features
import rasterio.shutil
import requests
import s3fs
from fdp import BUCKET, PREFIX, PROJ4, download, sciencebase_url
from pyproj import CRS

INPUTS = f'{BUCKET}/{PREFIX}/inputs'
MANIFEST = f'{INPUTS}/manifest.json'

# Stack order of the released R script; ranger draws mtry candidates by column
# position, so this order is part of the model specification.
PREDICTORS = {
    'floodplain': '100_yr_floodplain',
    'distwater': 'proximity_to_streams',
    'elevation': 'elevation',
    'hand': 'HAND',
    'nlcd': 'land_cover_2016',
    'pcurv': 'profile_curvature',
    'hydric': 'percent_hydric_soils',
    'twi': 'topographic_wetness_index',
    'channel': 'HUC12_channelization',
    'roads': '2016_urban_road_density',
    'm3dp': 'm3dp',
}
RESPONSE = {
    'presence': 'Input_ResponseVar_NOAA_reports_presence_100m.zip',
    'absence': 'Input_ResponseVar_absence_points_100m.zip',
}
HUCS = [f'{i:02d}' for i in range(1, 19)]
# The grid shared by 8 of the 11 predictors; the HUC raster and every
# training input are written on it.
REFERENCE = 'floodplain'
WBD = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/1/query'


def predictor_key(name: str) -> str:
    return f'{INPUTS}/predictors/{name}.tif'


def response_key(kind: str) -> str:
    return f'{INPUTS}/response/{kind}.csv'


HUC_KEY = f'{INPUTS}/huc2_100m.tif'
HUC_GEOJSON_KEY = f'{INPUTS}/huc2.geojson'


def s3_path(key: str) -> str:
    return f's3://{key}'


def read_manifest() -> dict:
    s3 = s3fs.S3FileSystem()
    if not s3.exists(MANIFEST):
        return {'predictors': {}, 'response': {}}
    with s3.open(MANIFEST, 'rb') as fh:
        return json.load(fh)


def write_manifest(manifest: dict) -> None:
    with s3fs.S3FileSystem().open(MANIFEST, 'w') as fh:
        json.dump(manifest, fh, indent=2)


def sha256(path: Path) -> str:
    with path.open('rb') as fh:
        return hashlib.file_digest(fh, 'sha256').hexdigest()


def grid_of(profile) -> dict:
    return {
        'shape': [profile['height'], profile['width']],
        'transform': list(profile['transform'])[:6],
        'crs': profile['crs'].to_wkt(),
    }


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------


def _fetch(zip_name: str, work: Path) -> tuple[Path, dict]:
    path = work / zip_name
    download(sciencebase_url(zip_name), path)
    return path, {'zip': zip_name, 'zip_bytes': path.stat().st_size, 'zip_sha256': sha256(path)}


def stage_predictor(name: str, work: Path) -> dict:
    zip_path, record = _fetch(f'Input_PredictorVar_CONUS_{PREDICTORS[name]}_100m.zip', work)
    with zipfile.ZipFile(zip_path) as zf:
        tifs = [
            n
            for n in zf.namelist()
            if n.lower().endswith('.tif') and not Path(n).name.startswith('._')
        ]
        if len(tifs) != 1:
            sys.exit(f'expected exactly one .tif in {zip_path.name}, found {tifs}')
        print(f'  extracting {tifs[0]}')
        zf.extract(tifs[0], work)
    zip_path.unlink()
    src = work / tifs[0]

    cog = work / f'{name}.tif'
    print(f'  writing COG ({src.stat().st_size / 1e9:.2f} GB source)')
    t0 = time.time()
    rasterio.shutil.copy(
        src,
        cog,
        driver='COG',
        COMPRESS='DEFLATE',
        PREDICTOR='YES',
        BIGTIFF='YES',
        BLOCKSIZE=512,
        OVERVIEWS='NONE',
        NUM_THREADS='ALL_CPUS',
    )
    print(f'    {cog.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s')

    with rasterio.open(src) as a, rasterio.open(cog) as b:
        if a.profile['dtype'] != b.profile['dtype'] or a.nodata != b.nodata:
            sys.exit(f'{name}: COG changed dtype or nodata')
        record |= {
            'source_tif': tifs[0],
            'source_bytes': src.stat().st_size,
            'dtype': b.profile['dtype'],
            'nodata': b.nodata,
            **grid_of(b.profile),
        }
    src.unlink()

    print(f'  uploading -> {s3_path(predictor_key(name))}')
    s3fs.S3FileSystem().put_file(str(cog), predictor_key(name))
    cog.unlink()
    return record


def stage_response(kind: str, work: Path) -> dict:
    zip_path, record = _fetch(RESPONSE[kind], work)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work / kind)
    shp = next((work / kind).rglob('*.shp'))
    # The .prj is ESRI WKT with an unnamed datum, so compare projection parameters.
    prj = CRS.from_wkt(shp.with_suffix('.prj').read_text())
    if CRS.from_proj4(prj.to_proj4()) != CRS.from_proj4(PROJ4):
        sys.exit(f'{shp.name}: unexpected projection {prj.to_proj4()}')

    con = duckdb.connect()
    con.sql('INSTALL spatial; LOAD spatial;')
    xy = con.execute(f"SELECT ST_X(geom) AS x, ST_Y(geom) AS y FROM ST_Read('{shp}')").fetchnumpy()
    local = work / f'{kind}.csv'
    np.savetxt(
        local,
        np.column_stack([xy['x'], xy['y']]),
        delimiter=',',
        header='x,y',
        comments='',
        fmt='%.4f',
    )
    s3fs.S3FileSystem().put_file(str(local), response_key(kind))
    print(f'  {len(xy["x"])} points -> {s3_path(response_key(kind))}')
    return record | {'points': int(len(xy['x']))}


def stage_hucs(grid: dict, work: Path) -> dict:
    params = {
        'where': "huc2 >= '01' AND huc2 <= '18'",
        'outFields': 'huc2,name',
        'outSR': 5070,
        'returnGeometry': 'true',
        'maxAllowableOffset': 50,
        'f': 'geojson',
    }
    r = requests.get(WBD, params=params, timeout=180)
    r.raise_for_status()
    features = sorted(r.json()['features'], key=lambda f: f['properties']['huc2'])
    if [f['properties']['huc2'] for f in features] != HUCS:
        sys.exit('WBD did not return HUC-2 01..18')
    geojson = work / 'huc2.geojson'
    geojson.write_bytes(r.content)

    print('  rasterizing HUC-2 onto the predictor grid')
    shape = tuple(grid['shape'])
    transform = rasterio.Affine(*grid['transform'])
    hucs = rasterio.features.rasterize(
        ((f['geometry'], int(f['properties']['huc2'])) for f in features),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype='uint8',
    )

    windows, cells = {}, {}
    for huc in HUCS:
        rows = np.flatnonzero((hucs == int(huc)).any(axis=1))
        cols = np.flatnonzero((hucs[rows[0] : rows[-1] + 1] == int(huc)).any(axis=0))
        windows[huc] = [
            int(rows[0]),
            int(cols[0]),
            int(rows[-1] - rows[0] + 1),
            int(cols[-1] - cols[0] + 1),
        ]
        cells[huc] = int((hucs[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] == int(huc)).sum())
        print(f'    {huc}: window {windows[huc]}, {cells[huc]:,} cells')

    tif = work / 'huc2_100m.tif'
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': 0,
        'count': 1,
        'height': shape[0],
        'width': shape[1],
        'transform': transform,
        'crs': rasterio.crs.CRS.from_wkt(grid['crs']),
    }
    with rasterio.MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(hucs, 1)
        with mem.open() as src:
            rasterio.shutil.copy(
                src,
                tif,
                driver='COG',
                COMPRESS='DEFLATE',
                BLOCKSIZE=512,
                OVERVIEWS='NONE',
                NUM_THREADS='ALL_CPUS',
            )

    s3 = s3fs.S3FileSystem()
    s3.put_file(str(geojson), HUC_GEOJSON_KEY)
    s3.put_file(str(tif), HUC_KEY)
    print(f'  -> {s3_path(HUC_KEY)}')
    return {'source': r.url, 'windows': windows, 'cells': cells}


def cmd_stage(_) -> None:
    manifest = read_manifest()
    s3 = s3fs.S3FileSystem()
    with tempfile.TemporaryDirectory(prefix='fdp-inputs-') as tmp:
        work = Path(tmp)
        for name in PREDICTORS:
            if name in manifest['predictors'] and s3.exists(predictor_key(name)):
                print(f'{name}: staged')
                continue
            print(f'{name}:')
            manifest['predictors'][name] = stage_predictor(name, work)
            write_manifest(manifest)

        for kind in RESPONSE:
            if kind in manifest['response'] and s3.exists(response_key(kind)):
                print(f'{kind} points: staged')
                continue
            print(f'{kind} points:')
            manifest['response'][kind] = stage_response(kind, work)
            write_manifest(manifest)
            shutil.rmtree(work / kind, ignore_errors=True)

        manifest['grid'] = grid_of_record(manifest['predictors'][REFERENCE])
        for name, rec in manifest['predictors'].items():
            rec['offset'] = offset(rec, manifest['grid'])
            if rec['offset'] != [0, 0]:
                print(f'{name}: offset {rec["offset"]} cells into the {REFERENCE} grid')

        if 'huc2' not in manifest or not s3.exists(HUC_KEY):
            print('huc2:')
            manifest['huc2'] = stage_hucs(manifest['grid'], work)
            write_manifest(manifest)
        else:
            print('huc2: staged')
    write_manifest(manifest)
    print()
    cmd_verify(None)


def grid_of_record(record: dict) -> dict:
    return {k: record[k] for k in ('shape', 'transform', 'crs')}


def offset(record: dict, grid: dict, tol: float = 1e-3) -> list[int]:
    """[row, col] of `grid`'s origin in `record`'s grid, which must contain it.

    The predictors share one 100 m lattice but not one extent: TWI comes on the
    larger grid of the published output, and profile curvature's origin carries
    ~2e-5 m of float noise. Reading each at its offset aligns them exactly.
    """
    a, b = record['transform'], grid['transform']
    if a[0] != b[0] or a[4] != b[4] or a[1] or a[3]:
        sys.exit(f'{record}: resolution differs from the reference grid')
    if CRS.from_proj4(CRS.from_wkt(record['crs']).to_proj4()) != CRS.from_proj4(PROJ4):
        sys.exit(f'{record["source_tif"]}: unexpected CRS')
    col, row = (b[2] - a[2]) / a[0], (b[5] - a[5]) / a[4]
    if abs(col - round(col)) > tol or abs(row - round(row)) > tol:
        sys.exit(f'{record["source_tif"]}: off the reference lattice by ({row}, {col}) cells')
    row, col = round(row), round(col)
    (h, w), (H, W) = record['shape'], grid['shape']
    if row < 0 or col < 0 or row + H > h or col + W > w:
        sys.exit(f'{record["source_tif"]}: does not cover the reference grid')
    return [row, col]


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def check() -> list[str]:
    s3 = s3fs.S3FileSystem()
    if not s3.exists(MANIFEST):
        return ['manifest.json missing']
    manifest = read_manifest()
    fails = []
    grid = manifest.get('grid')
    if grid is None:
        fails.append('manifest has no grid')

    for name in PREDICTORS:
        rec = manifest['predictors'].get(name)
        if rec is None:
            fails.append(f'{name}: not in manifest')
            continue
        with rasterio.open(s3_path(predictor_key(name))) as src:
            got = grid_of(src.profile)
            print(
                f'{name}: {src.profile["dtype"]} nodata={src.nodata} '
                f'shape={got["shape"]} blocks={src.block_shapes[0]}'
            )
            if grid and rec.get('offset') != offset(got | {'source_tif': name}, grid):
                fails.append(f'{name}: grid differs from manifest grid')
            if src.profile['dtype'] != rec['dtype'] or src.nodata != rec['nodata']:
                fails.append(f'{name}: dtype/nodata differ from manifest')

    for kind in RESPONSE:
        rec = manifest['response'].get(kind)
        if rec is None or not s3.exists(response_key(kind)):
            fails.append(f'{kind} points missing')
            continue
        with s3.open(response_key(kind)) as fh:
            n = sum(1 for _ in fh) - 1
        print(f'{kind}: {n} points')
        if n != rec['points']:
            fails.append(f'{kind}: {n} rows, manifest says {rec["points"]}')

    if 'huc2' not in manifest or not s3.exists(HUC_KEY):
        fails.append('huc2 raster missing')
    else:
        with rasterio.open(s3_path(HUC_KEY)) as src:
            if grid and grid_of(src.profile)['transform'] != grid['transform']:
                fails.append('huc2 raster grid differs from predictors')
        if sorted(manifest['huc2']['windows']) != HUCS:
            fails.append('huc2 windows incomplete')
    return fails


def cmd_verify(_) -> None:
    print(f'verifying s3://{INPUTS}')
    fails = check()
    if fails:
        print('\nFAIL')
        for f in fails:
            print(f'  - {f}')
        sys.exit(1)
    print('\nOK')


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('stage', help='download, convert and upload any inputs not yet staged')
    sub.add_parser('verify', help='check the staged inputs against the manifest')
    args = p.parse_args()
    {'stage': cmd_stage, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
