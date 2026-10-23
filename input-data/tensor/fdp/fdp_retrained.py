# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "rasterio",
#   "s3fs>=2024.0.0",
#   "topozarr",
#   "xarray",
#   "xproj",
#   "zarr>=3",
# ]
# ///
"""Mosaic retrained FDP watersheds and their attributions into a served store.

Reads each watershed's `fdp.tif`, `contributions.tif` and `contributions.json`
from a `fdp_train.py task --contributions` run and writes one multiscale store
on the published surface's grid, cropped to the watersheds included:

  FDP/processed/retrained/<run>-<distance>[-quick]/{0..7}/
    fdp              (y, x)          float32  probability
    contribution     (driver, y, x)  int16    x 1e-4; per cell, the watershed's
                                              baseline + sum over drivers = fdp
    dominant_driver  (y, x)          uint8    index of the largest contribution

`fdp` and `contribution` coarsen by mean, which keeps the sum identity at every
level up to int16 truncation. `dominant_driver` is recomputed per level from
that level's contributions, so it always agrees with them. Root attributes carry
the driver names, the per-watershed baselines and each driver's mean absolute
contribution over the store's cells.

Usage:
  python fdp_retrained.py build --run noaa-200612-202512 --distance planar --hucs 03
  python fdp_retrained.py verify --run noaa-200612-202512 --distance planar
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
import s3fs
import xarray as xr
import xproj  # noqa: F401 — registers the .proj accessor
import zarr
from fdp import BUCKET, CRS, LEVELS, OUT, STORE_KEY, VARIABLE, _bounds, _sync, _write_manifest
from fdp_inputs import HUCS
from fdp_points import RUNS
from fdp_train import model_prefix
from topozarr.coarsen import create_pyramid

CONTRIBUTION = 'contribution'
DOMINANT = 'dominant_driver'
SCALE = 1e-4
NODATA = -32768
NO_DRIVER = 255


def store_name(run: str, distance: str, quick: bool) -> str:
    return f'{run}-{distance}{"-quick" if quick else ""}'


def store_key(run: str, distance: str, quick: bool) -> str:
    return f'{BUCKET}/{OUT}/retrained/{store_name(run, distance, quick)}'


def published_transform() -> rasterio.Affine:
    store = zarr.storage.FsspecStore.from_url(f's3://{STORE_KEY}', read_only=True)
    return rasterio.Affine(*zarr.open_group(store, mode='r').attrs['spatial:transform'])


def cell_offset(t: rasterio.Affine, grid: rasterio.Affine) -> tuple[int, int]:
    """Row and column of `t`'s origin on `grid`, which must share its lattice."""
    if (t.a, t.e) != (grid.a, grid.e):
        raise ValueError(f'resolution {(t.a, t.e)} differs from the grid {(grid.a, grid.e)}')
    row, col = (t.f - grid.f) / grid.e, (t.c - grid.c) / grid.a
    if abs(row - round(row)) > 1e-6 or abs(col - round(col)) > 1e-6:
        raise ValueError(f'origin {(t.c, t.f)} is off the grid lattice')
    return round(row), round(col)


def read_watershed(prefix: str, grid: rasterio.Affine) -> dict:
    s3 = s3fs.S3FileSystem()
    meta = json.loads(s3.cat(f'{prefix}/contributions.json'))
    with rasterio.open(f's3://{prefix}/fdp.tif') as src:
        fdp = src.read(1, masked=True).filled(np.nan).astype('float32')
        offset = cell_offset(src.transform, grid)
    with rasterio.open(f's3://{prefix}/contributions.tif') as src:
        contribution = src.read()
        if src.nodata != NODATA or contribution.shape[0] != len(meta['drivers']):
            sys.exit(f'{prefix}/contributions.tif: unexpected nodata or band count')
    return {'fdp': fdp, 'contribution': contribution, 'offset': offset, 'meta': meta}


def mosaic(run: str, distance: str, quick: bool, hucs: list[str]) -> xr.Dataset:
    grid = published_transform()
    sheds = {}
    for huc in hucs:
        prefix = model_prefix(run, distance, quick, huc)
        print(f'  reading {prefix}')
        sheds[huc] = read_watershed(prefix, grid)
    drivers = sheds[hucs[0]]['meta']['drivers']
    if any(s['meta']['drivers'] != drivers for s in sheds.values()):
        sys.exit('watersheds disagree on driver order')

    r0 = min(s['offset'][0] for s in sheds.values())
    c0 = min(s['offset'][1] for s in sheds.values())
    r1 = max(s['offset'][0] + s['fdp'].shape[0] for s in sheds.values())
    c1 = max(s['offset'][1] + s['fdp'].shape[1] for s in sheds.values())
    fdp = np.full((r1 - r0, c1 - c0), np.nan, dtype='float32')
    contribution = np.full((len(drivers), r1 - r0, c1 - c0), NODATA, dtype='int16')
    for s in sheds.values():
        r, c = s['offset'][0] - r0, s['offset'][1] - c0
        h, w = s['fdp'].shape
        valid = np.isfinite(s['fdp'])
        fdp[r : r + h, c : c + w][valid] = s['fdp'][valid]
        contribution[:, r : r + h, c : c + w][:, valid] = s['contribution'][:, valid]

    valid = contribution[0] != NODATA
    mean_abs = [float(np.abs(band[valid], dtype='float64').mean() * SCALE) for band in contribution]

    x = grid.c + grid.a * (np.arange(c0, c1) + 0.5)
    y = grid.f + grid.e * (np.arange(r0, r1) + 0.5)
    ds = xr.Dataset(
        {
            VARIABLE: (('y', 'x'), fdp),
            CONTRIBUTION: (
                ('driver', 'y', 'x'),
                contribution,
                {'_FillValue': NODATA, 'scale_factor': SCALE, 'drivers': drivers},
            ),
        },
        coords={'x': x, 'y': y, 'driver': np.arange(len(drivers), dtype='int8')},
        attrs={
            'drivers': drivers,
            'baselines': {huc: s['meta']['baseline'] for huc, s in sheds.items()},
            'mean_abs_contribution': mean_abs,
            'attribution': sheds[hucs[0]]['meta']['method'],
            'run': run,
            'distance': distance,
            'hucs': hucs,
        },
    )
    return ds.proj.assign_crs(spatial_ref=CRS)


def write_dominant(local: Path, drivers: list[str]) -> None:
    """Add each level's dominant driver, chunked and sharded like its fdp."""
    for level in range(LEVELS):
        group = zarr.open_group(str(local), mode='r+', path=f'/{level}')
        contribution = group[CONTRIBUTION][:]
        valid = contribution[0] != NODATA
        dominant = np.where(valid, contribution.argmax(axis=0), NO_DRIVER).astype('uint8')
        like = group[VARIABLE]
        out = group.create_array(
            DOMINANT,
            shape=dominant.shape,
            dtype='uint8',
            chunks=like.chunks,
            shards=like.shards,
            fill_value=NO_DRIVER,
            dimension_names=['y', 'x'],
            attributes={
                'flag_values': list(range(len(drivers))),
                'flag_meanings': ' '.join(drivers),
                'long_name': 'predictor with the largest contribution',
            },
        )
        out[:] = dominant


def cmd_build(args) -> None:
    hucs = sorted(args.hucs or HUCS)
    name = store_name(args.run, args.distance, args.quick)
    print(f'mosaicking {len(hucs)} watershed(s) -> {name}')
    ds = mosaic(args.run, args.distance, args.quick, hucs)
    print(f'  window {dict(ds.sizes)}')

    print(f'building pyramid ({LEVELS} levels)')
    pyr = create_pyramid(ds, levels=LEVELS, x_dim='x', y_dim='y', method='mean')
    key = store_key(args.run, args.distance, args.quick)
    with tempfile.TemporaryDirectory(prefix='fdp-retrained-') as tmp:
        local = Path(tmp) / name
        t0 = time.time()
        pyr.write(str(local), mode='w')
        zarr.open_group(str(local), mode='r+').attrs.update(ds.attrs)
        write_dominant(local, ds.attrs['drivers'])
        mb = sum(f.stat().st_size for f in local.rglob('*') if f.is_file()) / 1e6
        print(f'  {mb:.1f} MB in {time.time() - t0:.0f}s')
        print(f'syncing -> s3://{key}')
        _sync(local, key, name)

    _write_manifest(
        {
            'id': f'retrained/{name}',
            'variable': VARIABLE,
            'variables': [VARIABLE, CONTRIBUTION, DOMINANT],
            'drivers': ds.attrs['drivers'],
            'baselines': ds.attrs['baselines'],
            'hucs': hucs,
            'units': 'probability',
            'crs': CRS,
            'bounds': _bounds(ds),
            'levels': LEVELS,
            'resolution_m': 100,
            'source': f's3://{RUNS}/{args.run}',
        }
    )
    print()
    verify(args.run, args.distance, args.quick)


def verify(run: str, distance: str, quick: bool) -> None:
    """Check levels, value ranges, the sum identity and the dominant driver."""
    key = store_key(run, distance, quick)
    store = zarr.storage.FsspecStore.from_url(f's3://{key}', read_only=True)
    root = zarr.open_group(store, mode='r')
    drivers, baselines = root.attrs['drivers'], root.attrs['baselines']
    fails = []
    for level in range(LEVELS):
        group = zarr.open_group(store, mode='r', path=f'/{level}')
        fdp = group[VARIABLE]
        ny, nx = fdp.shape
        y0, x0 = max(0, (ny - 1024) // 2), max(0, (nx - 1024) // 2)
        win = (slice(y0, y0 + 1024), slice(x0, x0 + 1024))
        p = np.asarray(fdp[win])
        c = np.asarray(group[CONTRIBUTION][(slice(None), *win)])
        d = np.asarray(group[DOMINANT][win])
        valid = np.isfinite(p)
        if not valid.any():
            fails.append(f'level {level}: sample window is empty')
            continue
        if (c[0][valid] == NODATA).any() or (d[valid] == NO_DRIVER).any():
            fails.append(f'level {level}: attribution missing where fdp is set')
        total = c[:, valid].astype('float64').sum(axis=0) * SCALE
        # every watershed's baseline is within a few 1e-3 of 0.5, so one bound covers the mosaic
        residual = np.abs(p[valid] - total - np.mean(list(baselines.values())))
        spread = np.ptp(list(baselines.values())) if len(baselines) > 1 else 0.0
        tol = len(drivers) * SCALE * (level + 1) + spread + 1e-6
        if residual.max() > tol:
            fails.append(f'level {level}: sum identity off by {residual.max():.2e} (> {tol:.1e})')
        if (d[valid] != c[:, valid].argmax(axis=0)).any():
            fails.append(f'level {level}: dominant_driver disagrees with contribution')
        print(
            f'/{level}: shape={fdp.shape} fdp {np.nanmin(p):.3f}..{np.nanmax(p):.3f}'
            f' max residual {residual.max():.1e}'
        )
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
    for name in ('build', 'verify'):
        s = sub.add_parser(name)
        s.add_argument('--run', required=True)
        s.add_argument('--distance', choices=['lonlat', 'planar'], default='planar')
        s.add_argument('--quick', action='store_true')
        if name == 'build':
            s.add_argument('--hucs', nargs='+', choices=HUCS)
    args = p.parse_args()
    if args.cmd == 'build':
        cmd_build(args)
    else:
        verify(args.run, args.distance, args.quick)


if __name__ == '__main__':
    main()
