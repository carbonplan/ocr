# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "rasterio",
#   "s3fs>=2024.0.0",
#   "zarr>=3",
# ]
# ///
"""Train and predict the FDP random forest for HUC-2 watersheds.

`task` runs one watershed on the current machine: it window-reads that
watershed's predictors from the staged COGs, masks them to its HUC raster
cells (cached under $FDP_CACHE/predictors/), runs train_fdp.R, compares the
prediction with the published surface, and uploads everything to

  FDP/runs/<run>/<distance>[-quick]/huc<NN>/

`run` names a points set built by fdp_points.py. `--distance lonlat` keeps the
released script's spatial-sorting distances; `planar` corrects them (see
train_fdp.R). `--quick` shrinks tuning and the forest for a smoke test.
`--contributions` adds per-cell attributions of the prediction to each
predictor (contrib_fdp.R).

`launch` submits one `task` per watershed as a Coiled batch job. dismo has no
linux-aarch64 build, so the VMs are x86.

The R packages are not in any pixi environment: locally, point `--rscript` at
an R with them installed (see the README); on Coiled they are added to the
synced environment as conda extras.

Usage:
  python fdp_train.py task --run original --huc 03 [--rscript PATH] [--quick] [--contributions]
  python fdp_train.py launch --run original --hucs 03 [--vm-type m7i.4xlarge] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import s3fs
import zarr
from fdp import CACHE, VARIABLE, store_uri
from fdp_inputs import HUC_KEY, HUCS, PREDICTORS, predictor_key, read_manifest
from fdp_points import RUNS, run_key
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent
R_SCRIPT = 'train_fdp.R'
CONTRIB_SCRIPT = 'contrib_fdp.R'
CONTRIB_SOURCE = 'fdp_contrib.cpp'
R_PACKAGES = [
    'compilers',
    'r-rcpp',
    'r-base',
    'r-raster',
    'r-terra',
    'r-sp',
    'r-dismo',
    'r-ranger',
    'r-tuneranger',
    'r-mlr',
    'r-data.table',
    'r-jsonlite',
    'r-snow',
]
QUICK = {'iters': 3, 'trees': 300}


def model_prefix(run: str, distance: str, quick: bool, huc: str) -> str:
    return f'{RUNS}/{run}/{distance}{"-quick" if quick else ""}/huc{huc}'


def r_script(name: str = R_SCRIPT) -> Path:
    for path in (HERE / name, Path.cwd() / name):
        if path.exists():
            return path
    sys.exit(f'{name} not found next to {__file__} or in {Path.cwd()}')


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------


def prepare_predictors(huc: str, manifest: dict) -> tuple[Path, np.ndarray]:
    """Write the watershed's predictors, cells outside it set to nodata.

    The files are cached per watershed and reused while the staged inputs they
    were cut from are unchanged. Returns their directory and the watershed's
    cell mask over its window.
    """
    grid = manifest['grid']
    offsets = {name: manifest['predictors'][name]['offset'] for name in PREDICTORS}
    window = manifest['huc2']['windows'][huc]
    r0, c0, h, w = window
    win = Window(c0, r0, w, h)
    with rasterio.open(f's3://{HUC_KEY}') as src:
        outside = src.read(1, window=win) != int(huc)

    dest = CACHE / 'predictors' / f'huc{huc}'
    stamp = {
        'window': window,
        'grid': grid,
        'huc2': manifest['huc2']['source'],
        'predictors': {
            name: {k: manifest['predictors'][name][k] for k in ('zip_sha256', 'offset')}
            for name in PREDICTORS
        },
    }
    if (dest / 'inputs.json').exists() and json.loads((dest / 'inputs.json').read_text()) == stamp:
        print(f'  cached: {dest}', flush=True)
        return dest, ~outside
    if dest.exists():
        shutil.rmtree(dest)
    partial = dest.with_name(dest.name + '.partial')
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    transform = rasterio.windows.transform(win, rasterio.Affine(*grid['transform']))
    for name in PREDICTORS:
        t0 = time.time()
        dr, dc = offsets[name]
        with rasterio.open(f's3://{predictor_key(name)}') as src:
            if src.nodata is None:
                sys.exit(f'{name} has no nodata value')
            data = src.read(1, window=Window(c0 + dc, r0 + dr, w, h))
            data[outside] = src.nodata
            profile = src.profile | {
                'driver': 'GTiff',
                'height': h,
                'width': w,
                'transform': transform,
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'compress': 'deflate',
                'BIGTIFF': 'IF_SAFER',
            }
        with rasterio.open(partial / f'{name}.tif', 'w', **profile) as out:
            out.write(data, 1)
        print(f'  {name}: {h}x{w} in {time.time() - t0:.0f}s', flush=True)
    (partial / 'inputs.json').write_text(json.dumps(stamp, indent=2))
    partial.rename(dest)
    return dest, ~outside


class Monitor(threading.Thread):
    """Poll peak memory of a process group and peak disk use of a directory."""

    def __init__(self, pgid: int, path: Path, every: float = 10.0):
        super().__init__(daemon=True)
        self.pgid, self.path, self.every = pgid, path, every
        self.peak_rss_gb = 0.0
        self.peak_disk_gb = 0.0
        self.done = threading.Event()

    def rss(self) -> float:
        out = subprocess.run(
            ['ps', '-A', '-o', 'pgid=,rss='], capture_output=True, text=True
        ).stdout
        kb = sum(
            int(r) for g, r in (line.split() for line in out.splitlines()) if int(g) == self.pgid
        )
        return kb / 1e6

    def disk(self) -> float:
        return sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file()) / 1e9

    def run(self) -> None:
        while not self.done.wait(self.every):
            self.peak_rss_gb = max(self.peak_rss_gb, self.rss())
            self.peak_disk_gb = max(self.peak_disk_gb, self.disk())


def run_r(cmd: list[str], log: Path, work: Path) -> dict:
    """Run an R script, teeing its output to `log`; returns time and peaks."""
    t0 = time.time()
    with log.open('w') as fh:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
        )
        monitor = Monitor(proc.pid, work)
        monitor.start()
        for line in proc.stdout:
            if 'codetools' in line:
                continue
            fh.write(line)
            print(line, end='', flush=True)
        code = proc.wait()
        monitor.done.set()
    if code:
        sys.exit(f'{Path(cmd[1]).name} exited {code}; see {log}')
    return {
        'seconds': round(time.time() - t0),
        'peak_rss_gb': round(monitor.peak_rss_gb, 1),
        'peak_disk_gb': round(monitor.peak_disk_gb, 1),
    }


def agreement(a: np.ndarray, b: np.ndarray, sample: int, seed: int) -> dict:
    d = np.abs(a - b)
    rng = np.random.default_rng(seed)
    pick = rng.choice(a.size, size=min(sample, a.size), replace=False)
    spearman = pd.Series(a[pick]).rank().corr(pd.Series(b[pick]).rank())
    top_a, top_b = a >= np.quantile(a, 0.8), b >= np.quantile(b, 0.8)
    return {
        'cells': int(a.size),
        'mean': {'ours': float(a.mean()), 'published': float(b.mean())},
        'pearson': float(np.corrcoef(a, b)[0, 1]),
        'spearman_sample': float(spearman),
        'abs_diff': {
            'mean': float(d.mean()),
            'p50': float(np.median(d)),
            'p90': float(np.quantile(d, 0.9)),
            'share_gt_0.1': float((d > 0.1).mean()),
            'share_gt_0.25': float((d > 0.25).mean()),
        },
        'top20_overlap': float((top_a & top_b).sum() / top_b.sum()),
    }


def compare(
    prediction: Path,
    inside: np.ndarray,
    floodplain: Path,
    sample: int = 2_000_000,
    seed: int = 0,
) -> dict:
    """Agreement with the published surface over the watershed's cells.

    The 100-yr floodplain holds no pseudo-absences, so the model scores it near
    1 throughout; `outside_floodplain` measures agreement where the surface
    actually ranks cells.
    """
    with rasterio.open(prediction) as src:
        ours = src.read(1, masked=True)
        t = src.transform
    with rasterio.open(floodplain) as src:
        in_floodplain = src.read(1) == 1
    root = zarr.open_group(
        zarr.storage.FsspecStore.from_url(f'{store_uri()}/0', read_only=True), mode='r'
    )
    xs, ys = np.asarray(root['x'][:]), np.asarray(root['y'][:])
    c0 = int(np.flatnonzero(np.isclose(xs, t.c + t.a / 2))[0])
    r0 = int(np.flatnonzero(np.isclose(ys, t.f + t.e / 2))[0])
    published = np.asarray(root[VARIABLE][r0 : r0 + ours.shape[0], c0 : c0 + ours.shape[1]])

    mine, theirs = ~ours.mask & inside, np.isfinite(published) & inside
    both = mine & theirs
    a = ours.data[both].astype('float64')
    b = published[both].astype('float64')
    fp = in_floodplain[both]
    return {
        'cells_huc': int(inside.sum()),
        'cells_both': int(both.sum()),
        'cells_ours_only': int((mine & ~theirs).sum()),
        'cells_published_only': int((theirs & ~mine).sum()),
        **agreement(a, b, sample, seed),
        'floodplain_share': float(fp.mean()),
        'outside_floodplain': agreement(a[~fp], b[~fp], sample, seed),
    }


def upload(local: Path, prefix: str) -> None:
    if '/runs/' not in prefix or not prefix.rsplit('/', 1)[-1].startswith('huc'):
        raise ValueError(f'refusing to write to {prefix!r}')
    s3 = s3fs.S3FileSystem()
    if s3.exists(prefix):
        s3.rm(prefix, recursive=True)
    for f in sorted(local.iterdir()):
        s3.put_file(str(f), f'{prefix}/{f.name}')
    print(f'uploaded {len(list(local.iterdir()))} files -> s3://{prefix}')


def cmd_task(args) -> None:
    huc = args.huc or os.environ.get('COILED_BATCH_TASK_INPUT')
    if huc not in HUCS:
        sys.exit(f'--huc must be one of {HUCS}, got {huc!r}')
    prefix = model_prefix(args.run, args.distance, args.quick, huc)
    threads = args.threads or os.cpu_count()
    settings = QUICK if args.quick else {'iters': 70, 'trees': 1000}
    print(f'HUC {huc} -> s3://{prefix} ({threads} threads, {settings})', flush=True)

    work = Path(tempfile.mkdtemp(prefix=f'fdp-huc{huc}-', dir=args.work))
    out, tmp = work / 'out', work / 'tmp'
    out.mkdir()
    tmp.mkdir()

    t0 = time.time()
    manifest = read_manifest()
    predictors, inside = prepare_predictors(huc, manifest)
    inputs = {'window': manifest['huc2']['windows'][huc], 'cells': int(inside.sum())}
    s3fs.S3FileSystem().get_file(run_key(args.run, 'points.csv'), str(work / 'points.csv'))
    prepare_seconds = round(time.time() - t0)

    cmd = [
        args.rscript,
        str(r_script()),
        '--huc',
        huc,
        '--predictors',
        str(predictors),
        '--points',
        str(work / 'points.csv'),
        '--out',
        str(out),
        '--tmp',
        str(tmp),
        '--threads',
        str(threads),
        '--iters',
        str(settings['iters']),
        '--trees',
        str(settings['trees']),
        '--distance',
        args.distance,
    ]
    r = run_r(cmd, out / 'log.txt', work)
    contributions = None
    if args.contributions:
        print('attributing the prediction to its predictors', flush=True)
        contrib_cmd = [
            args.rscript,
            str(r_script(CONTRIB_SCRIPT)),
            '--model',
            str(out / 'rf.rds'),
            '--train',
            str(out / 'train.rds'),
            '--predictors',
            str(predictors),
            '--prediction',
            str(out / 'fdp.tif'),
            '--out',
            str(out),
            '--threads',
            str(threads),
        ]
        contributions = run_r(contrib_cmd, out / 'contrib-log.txt', work)

    print('comparing with the published surface', flush=True)
    stats = compare(out / 'fdp.tif', inside, predictors / 'floodplain.tif')
    (out / 'compare.json').write_text(json.dumps(stats, indent=2))
    (out / 'run.json').write_text(
        json.dumps(
            {
                'huc': huc,
                'run': args.run,
                'distance': args.distance,
                'quick': args.quick,
                'settings': settings,
                'threads': threads,
                'inputs': inputs,
                'prepare_seconds': prepare_seconds,
                'r': r,
                'contributions': contributions,
                'host': {
                    'platform': platform.platform(),
                    'cpus': os.cpu_count(),
                    'vm_type': os.environ.get('COILED_VM_TYPE'),
                },
            },
            indent=2,
        )
    )
    upload(out, prefix)
    if not args.keep:
        shutil.rmtree(work)


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


def cmd_launch(args) -> None:
    hucs = args.hucs or HUCS
    cmd = [
        'coiled',
        'batch',
        'run',
        '--name',
        f'fdp-{args.run}-{args.distance}{"-quick" if args.quick else ""}',
        '--region',
        'us-west-2',
        '--vm-type',
        args.vm_type,
        '--disk-size',
        str(args.disk_size),
        '--tag',
        'Project=OCR',
        '--forward-aws-credentials',
        '--spot-policy',
        'spot_with_fallback' if args.spot else 'on-demand',
        '--map-over-values',
        ','.join(hucs),
        '--upload',
        str(HERE / 'fdp.py'),
        '--upload',
        str(HERE / 'fdp_inputs.py'),
        '--upload',
        str(HERE / 'fdp_points.py'),
        '--upload',
        str(HERE / R_SCRIPT),
        '--upload',
        str(HERE / CONTRIB_SCRIPT),
        '--upload',
        str(HERE / CONTRIB_SOURCE),
    ]
    for pkg in R_PACKAGES:
        cmd += ['--package-sync-conda-extras', pkg]
    cmd += [
        '--',
        'python',
        str(HERE / 'fdp_train.py'),
        'task',
        '--run',
        args.run,
        '--distance',
        args.distance,
    ]
    if args.quick:
        cmd.append('--quick')
    if args.contributions:
        cmd.append('--contributions')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--run', required=True, help='points set from fdp_points.py')
    common.add_argument('--distance', choices=['lonlat', 'planar'], default='lonlat')
    common.add_argument('--quick', action='store_true')
    common.add_argument(
        '--contributions', action='store_true', help='also write per-cell predictor attributions'
    )

    t = sub.add_parser('task', parents=[common], help='train and predict one watershed here')
    t.add_argument('--huc', help='defaults to $COILED_BATCH_TASK_INPUT')
    t.add_argument('--rscript', default='Rscript')
    t.add_argument('--threads', type=int)
    t.add_argument('--work', help='parent directory for the working files')
    t.add_argument('--keep', action='store_true', help='keep the working directory')

    la = sub.add_parser('launch', parents=[common], help='submit one task per watershed to Coiled')
    la.add_argument('--hucs', nargs='+', choices=HUCS)
    la.add_argument('--vm-type', default='m7i.4xlarge')
    la.add_argument('--disk-size', type=int, default=200)
    la.add_argument('--spot', action='store_true')

    args = p.parse_args()
    {'task': cmd_task, 'launch': cmd_launch}[args.cmd](args)


if __name__ == '__main__':
    main()
