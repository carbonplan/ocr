# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "requests",
#   "rioxarray",
#   "s3fs>=2024.0.0",
#   "topozarr==0.1.2",
#   "xarray",
#   "xproj",
#   "zarr>=3",
# ]
# ///
"""Build the USGS Flood Damage Probability layer as a native-CRS topozarr pyramid.

Source: Collins et al. (2022), "Predicting flood damage probability across the
conterminous United States", USGS ScienceBase doi:10.5066/P954TTQN. One 100 m
CONUS raster of random-forest-predicted probabilities in [0, 1], NAD83 / Conus
Albers (EPSG:5070).

The pyramid stays in EPSG:5070 rather than being reprojected to EPSG:4326;
zarr-layer reprojects at render time from the store's proj4 + metre bounds.

`build` moves ~12 GB and holds a 6 GB array in memory, so run it on a VM next to
the bucket (see the README).

Usage:
  python fdp.py build
  python fdp.py verify
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import rioxarray  # noqa: F401 — registers open_rasterio / the .rio accessor
import s3fs
import xproj  # noqa: F401 — registers the .proj accessor
import zarr
from topozarr.coarsen import create_pyramid

BUCKET = 'carbonplan-ocr'
PREFIX = 'ocr-explore/FDP'
RAW = f'{PREFIX}/raw'
OUT = f'{PREFIX}/processed'

NAME = 'flood_damage_probability'
VARIABLE = 'fdp'
CRS = 'EPSG:5070'
PROJ4 = (
    '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 '
    '+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'
)
LEVELS = 8  # 100 m (L0) .. ~12.8 km (L7); the coarse levels are tiny
CLIM = [0.0, 1.0]
DOI = 'https://doi.org/10.5066/P954TTQN'

SB_ITEM = '6170694ed34ea36449a67ef7'
SB_ZIP = 'Output_CONUS_FDP_100m.zip'
SB_URL = f'https://www.sciencebase.gov/catalog/file/get/{SB_ITEM}?name={SB_ZIP}'
TIF_NAME = 'CONUS_FDP_100m.tif'

CACHE = Path(os.environ.get('FDP_CACHE', Path.home() / '.cache/ocr/fdp'))
RAW_KEY = f'{BUCKET}/{RAW}/{TIF_NAME}'
STORE_KEY = f'{BUCKET}/{OUT}/{NAME}'


def raw_uri() -> str:
    return f's3://{RAW_KEY}'


def store_uri() -> str:
    return f's3://{STORE_KEY}'


# ---------------------------------------------------------------------------
# Source raster
# ---------------------------------------------------------------------------


def _download_zip(dest: Path) -> None:
    """Stream the 3 GB source zip from ScienceBase.

    The endpoint ignores Range headers and streams the whole file, so there is
    no way to pull just the tif member: the zip lands on disk in full.
    """
    part = dest.with_suffix(dest.suffix + '.part')
    print(f'  downloading {SB_ZIP}')
    t0 = time.time()
    # A captured log has no carriage return, so tick every ~320 MB there instead
    # of redrawing one line every chunk.
    tty = sys.stdout.isatty()
    with requests.get(SB_URL, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        got = 0
        with part.open('wb') as fh:
            for i, chunk in enumerate(r.iter_content(chunk_size=32 * 1024 * 1024)):
                fh.write(chunk)
                got += len(chunk)
                if tty:
                    print(f'    {got / 1e9:5.2f} GB', end='\r', flush=True)
                elif i % 10 == 0:
                    print(f'    {got / 1e9:5.2f} GB', flush=True)
    part.rename(dest)
    dt = time.time() - t0
    print(f'    {got / 1e9:.2f} GB in {dt:.0f}s ({got / 1e6 / (dt + 1e-9):.1f} MB/s)')


def _extract_tif(zip_path: Path, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + '.part')
    with zipfile.ZipFile(zip_path) as zf:
        tifs = [
            n
            for n in zf.namelist()
            if n.lower().endswith('.tif') and not Path(n).name.startswith('._')
        ]
        if len(tifs) != 1:
            sys.exit(f'expected exactly one .tif in {zip_path.name}, found {tifs}')
        print(f'  extracting {tifs[0]}')
        with zf.open(tifs[0]) as src, part.open('wb') as out:
            shutil.copyfileobj(src, out, length=32 * 1024 * 1024)
    part.rename(dest)
    print(f'    {dest.stat().st_size / 1e9:.2f} GB')


def source_tif() -> Path:
    """Return the source GeoTIFF locally, staging it in S3 the first time.

    Once the raw copy is in the bucket a rebuild pulls it from there, so the slow
    USGS download happens once ever rather than once per build.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    tif = CACHE / TIF_NAME
    if tif.exists():
        print(f'  cached: {tif} ({tif.stat().st_size / 1e9:.2f} GB)')
        return tif

    s3 = s3fs.S3FileSystem()
    if s3.exists(RAW_KEY):
        print(f'  pulling {raw_uri()}')
        t0 = time.time()
        s3.get_file(RAW_KEY, str(tif))
        print(f'    in {time.time() - t0:.0f}s')
        return tif

    zip_path = CACHE / SB_ZIP
    if not zip_path.exists():
        _download_zip(zip_path)
    _extract_tif(zip_path, tif)
    print(f'  staging -> {raw_uri()}')
    t0 = time.time()
    s3.put_file(str(tif), RAW_KEY)
    print(f'    in {time.time() - t0:.0f}s')
    return tif


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def open_native(tif: Path):
    """Open the GeoTIFF on its native grid (metre Albers x/y), nodata masked to NaN."""
    print(f'  opening {tif}')
    da = rioxarray.open_rasterio(tif, masked=True)
    da = da.squeeze('band', drop=True).astype('float32')
    da = da.drop_vars('spatial_ref', errors='ignore')
    # The source band attrs carry STATISTICS_MEAN/STDDEV=nan, which topozarr writes
    # verbatim into the array zarr.json as a bare `NaN` token: valid to Python's
    # json but rejected by the browser's JSON.parse. We don't need them.
    da.attrs = {}
    ds = da.to_dataset(name=VARIABLE)
    # One sequential decode of the striped source into memory (~6 GB), so
    # Pyramid.write() then slices numpy regions with no repeated decode.
    print('  loading into memory ...')
    t0 = time.time()
    ds.load()
    print(f'    loaded in {time.time() - t0:.1f}s; sizes={dict(ds.sizes)}')
    return ds.proj.assign_crs(spatial_ref=CRS)


def _bounds(ds) -> list[float]:
    x, y = ds['x'].values, ds['y'].values
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    return [
        round(float(x.min()) - dx / 2, 1),
        round(float(y.min()) - dy / 2, 1),
        round(float(x.max()) + dx / 2, 1),
        round(float(y.max()) + dy / 2, 1),
    ]


def _sync(local: Path, prefix: str) -> None:
    """Upload `local` to s3://<prefix>/, pruning keys this build did not write.

    The prune is what a previous build's orphans need: chunks left behind by a
    change in shape or chunking, which a reader could otherwise still pick up.
    It deletes under `prefix`, so the guard below keeps a wrong prefix from
    reaching anything but this store.
    """
    if not prefix.endswith(f'/{NAME}') or prefix.count('/') < 3:
        raise ValueError(f'refusing to sync to prefix {prefix!r}')

    s3 = s3fs.S3FileSystem()
    files = [f for f in local.rglob('*') if f.is_file()]
    written = {f.relative_to(local).as_posix() for f in files}

    def put(f: Path) -> None:
        s3.put_file(str(f), f'{prefix}/{f.relative_to(local).as_posix()}')

    with ThreadPoolExecutor(max_workers=16) as ex:
        for _ in ex.map(put, files):
            pass
    print(f'  {len(files)} objects uploaded', end='')

    stale = {k.removeprefix(f'{prefix}/') for k in s3.find(prefix)} - written
    if stale:
        s3.rm([f'{prefix}/{k}' for k in sorted(stale)])
        print(f', {len(stale)} stale removed', end='')
    print()


def _write_manifest(record: dict) -> None:
    """Merge `record` into FDP/processed/manifest.json (keyed by id)."""
    s3 = s3fs.S3FileSystem()
    key = f'{BUCKET}/{OUT}/manifest.json'
    existing = {}
    if s3.exists(key):
        with s3.open(key, 'rb') as fh:
            existing = {r['id']: r for r in json.load(fh).get('stores', [])}
    existing[record['id']] = record
    with s3.open(key, 'w') as fh:
        json.dump({'stores': sorted(existing.values(), key=lambda r: r['id'])}, fh, indent=2)
    print(f'manifest: {len(existing)} stores -> s3://{key}')


def cmd_build(_) -> None:
    ds = open_native(source_tif())

    print(f'building pyramid ({LEVELS} levels, native {CRS})')
    pyr = create_pyramid(ds, levels=LEVELS, x_dim='x', y_dim='y', method='mean')
    with tempfile.TemporaryDirectory(prefix='fdp-') as tmp:
        local = Path(tmp) / 'store'
        t0 = time.time()
        pyr.write(str(local), mode='w')
        mb = sum(f.stat().st_size for f in local.rglob('*') if f.is_file()) / 1e6
        print(f'  {mb:.1f} MB in {time.time() - t0:.0f}s')
        for lvl in range(LEVELS):
            arr = zarr.open_group(str(local), mode='r', path=f'/{lvl}')[VARIABLE]
            print(f'    /{lvl}: shape={arr.shape} chunks={arr.chunks}')

        print(f'syncing -> {store_uri()}')
        t0 = time.time()
        _sync(local, STORE_KEY)
        print(f'  uploaded in {time.time() - t0:.0f}s')

    _write_manifest(
        {
            'id': NAME,
            'variable': VARIABLE,
            'units': 'probability',
            'clim': CLIM,
            'crs': CRS,
            'proj4': PROJ4,
            'bounds': _bounds(ds),
            'levels': LEVELS,
            'resolution_m': 100,
            'source': DOI,
        }
    )

    print()
    cmd_verify(None)
    print(f'\ndone: {store_uri()}')
    print(f'https: https://{BUCKET}.s3.us-west-2.amazonaws.com/{OUT}/{NAME}')


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def _sample(arr) -> np.ndarray:
    """Read the whole level, or a centred 2048x2048 window when it is large."""
    if arr.size <= 4_000_000:
        return np.asarray(arr[:])
    ny, nx = arr.shape
    # clamp: a negative offset would wrap to the grid edge, which is all ocean
    y0, x0 = max(0, (ny - 2048) // 2), max(0, (nx - 2048) // 2)
    return np.asarray(arr[y0 : y0 + 2048, x0 : x0 + 2048])


def check(uri: str) -> list[str]:
    """Read a built store and return everything wrong with it."""
    remote = uri.startswith('s3://')
    store = zarr.storage.FsspecStore.from_url(uri, read_only=True) if remote else uri
    root = zarr.open_group(store, mode='r')
    fails = []

    code = dict(root.attrs).get('proj:code')
    print(f'proj:code={code}')
    if code != CRS:
        fails.append(f'proj:code is {code!r}, expected {CRS!r}')

    prev = None
    for level in range(LEVELS):
        try:
            arr = zarr.open_group(store, mode='r', path=f'/{level}')[VARIABLE]
        except (KeyError, zarr.errors.GroupNotFoundError):
            fails.append(f'level {level} missing')
            break
        d = _sample(arr)
        d = d[np.isfinite(d)]
        rng = f'min={d.min():.4f} max={d.max():.4f}' if d.size else 'all-NaN'
        print(f'/{level}: shape={arr.shape} chunks={arr.chunks} dtype={arr.dtype} {rng}')
        if str(arr.dtype) != 'float32':
            fails.append(f'level {level} dtype is {arr.dtype}, expected float32')
        if d.size and (d.min() < 0 or d.max() > 1):
            fails.append(f'level {level} outside [0, 1]: {d.min():.4f}..{d.max():.4f}')
        if prev is not None:
            want = tuple(p // 2 for p in prev)  # topozarr floors odd dimensions
            if tuple(arr.shape) != want:
                fails.append(f'level {level} shape {arr.shape}, expected {want} (half of {prev})')
        prev = tuple(arr.shape)

    if not remote:
        return fails

    s3 = s3fs.S3FileSystem()
    key = f'{BUCKET}/{OUT}/manifest.json'
    if not s3.exists(key):
        fails.append('manifest.json missing')
        return fails
    with s3.open(key, 'rb') as fh:
        rec = {r['id']: r for r in json.load(fh).get('stores', [])}.get(NAME)
    if rec is None:
        fails.append(f'no manifest record for {NAME}')
    else:
        print(f'manifest: bounds={rec["bounds"]} levels={rec["levels"]}')
        for k, want in (('crs', CRS), ('variable', VARIABLE), ('levels', LEVELS)):
            if rec.get(k) != want:
                fails.append(f'manifest {k} is {rec.get(k)!r}, expected {want!r}')
    return fails


def cmd_verify(_) -> None:
    print(f'verifying {store_uri()}')
    fails = check(store_uri())
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
    sub.add_parser('build', help='fetch the source if needed, build the pyramid, sync, verify')
    sub.add_parser('verify', help='check the served store and its manifest record')

    args = p.parse_args()
    {'build': cmd_build, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
