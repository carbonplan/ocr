"""Simona Meiler's published ERA5 event set, cropped to CONUS.

The frequency-corrected global hazard behind the published maps is one
1,395,323-storm catalogue (10 files x 40 members, 1981-2019) stored six
times, once per basin, each copy with that basin's frequency and winds only
on that basin's cells. `crop` keeps the columns that are CONUS cells and the
events with wind there, writes a CLIMADA hazard plus event and column tables
under published/conus/, and checks that CLIMADA's ranking on the crop
reproduces the Dryad points. `match` tests whether any of her 400 member
runs is one of the shared track files, by storm start dates.

Both run on a Coiled VM (`submit`), since the source file is 35 GB of
uncompressed, unchunked HDF5 and the laptop's link to S3 is slow. `tailset`
then turns the crop into the set, levels and events files `chaz_tail_test.py`
reads under the tag `published`, with the Dryad values as the levels.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chaz_events as ce  # noqa: E402

PUB = f'{ce.PREFIX}/published'
HAZ_KEY = f'{PUB}/TC_global_0300as_CHAZ_ERA5_freq-corr.hdf5'
OUT = f'{PUB}/conus'
CROP_KEY = f'{OUT}/era5_conus.hdf5'
COLS_KEY = f'{OUT}/era5_conus_columns.parquet'
EVENTS_KEY = f'{OUT}/era5_conus_events.parquet'
CATALOGUE_KEY = f'{OUT}/era5_catalogue.parquet'
CHECK_KEY = f'{OUT}/era5_conus_check.parquet'
MATCH_KEY = f'{OUT}/track_match.parquet'
BASINS = ('EP', 'NA', 'NI', 'SI', 'SP', 'WP')

log = logging.getLogger('chaz_published')


def cell_key(lon, lat) -> np.ndarray:
    """Integer id of a 300 arcsec cell, shared by her centroids, the Dryad
    points and our CONUS centroids. The lattice sits on odd 24ths of a
    degree, so the key is in 24ths; rounding to 12ths merges neighbours."""
    return (
        np.round(np.asarray(lon) * 24).astype(np.int64) * 10_000_000
        + np.round(np.asarray(lat) * 24).astype(np.int64)
        + 5_000_000
    )


def _say(msg: str) -> None:
    print(f'{datetime.datetime.utcnow():%H:%M:%S} {msg}', flush=True)


# ---------------------------------------------------------------------------
# crop


def cmd_crop(args):
    import h5py
    import scipy.sparse as sp
    from climada.hazard import Centroids, Hazard

    s3 = ce._s3()
    src = Path(args.hazard)
    if not src.exists():
        t0 = time.time()
        s3.get(f'{ce.BUCKET}/{HAZ_KEY}', str(src))
        _say(f'fetched {src.stat().st_size / 2**30:.1f} GiB in {time.time() - t0:.0f}s')

    ours = Centroids.from_hdf5(ce._fetch(s3, ce.CENTROIDS_KEY))
    our_keys = np.unique(cell_key(ours.lon, ours.lat))
    _say(f'{ours.size:,} CONUS centroids, {our_keys.size:,} distinct cells')

    with h5py.File(src, 'r') as h:
        cent = pickle.loads(bytes(h['centroids/block1_values'][0]))
        xy = np.array([[g.x, g.y] for g in cent[:, 0]])
        region_id = cent[:, 1].astype('int64')
        on_land = cent[:, 2].astype('bool')
        colkey = cell_key(xy[:, 0], xy[:, 1])
        keep = np.isin(colkey, our_keys)
        newcol = np.full(colkey.size, -1, dtype=np.int64)
        newcol[keep] = np.arange(int(keep.sum()))
        _say(
            f'{int(keep.sum()):,} of her {colkey.size:,} columns are CONUS cells '
            f'({np.unique(colkey[keep]).size:,} distinct)'
        )

        ip = h['intensity/indptr'][:].astype(np.int64)
        n_ev = ip.size - 1
        assert n_ev % len(BASINS) == 0, n_ev
        n_block = n_ev // len(BASINS)
        names = h['event_name']
        basins = []
        for b in range(len(BASINS)):
            nm = names[b * n_block]
            basins.append((nm.decode() if isinstance(nm, bytes) else nm).split('_')[-1])
        assert tuple(basins) == BASINS, basins
        freq = h['frequency'][:]
        date = h['date'][:]
        orig = h['orig'][:]
        _say(
            f'{n_ev:,} rows = {len(BASINS)} basins x {n_block:,} storms; '
            f'basin frequencies {dict(zip(basins, freq[::n_block].round(9)))}'
        )

        ind = h['intensity/indices']
        dat = h['intensity/data']
        rows, cols, vals = [], [], []
        step = args.step
        t0 = time.time()
        for a in range(0, n_ev, step):
            b = min(a + step, n_ev)
            p0, p1 = int(ip[a]), int(ip[b])
            if p1 == p0:
                continue
            idx = ind[p0:p1]
            nc = newcol[idx]
            m = nc >= 0
            if not m.any():
                continue
            r = np.repeat(np.arange(a, b, dtype=np.int64), np.diff(ip[a : b + 1]))
            rows.append(r[m])
            cols.append(nc[m])
            vals.append(dat[p0:p1][m].astype(np.float32))
            if (a // step) % 5 == 0:
                _say(
                    f'  rows {a:,}-{b:,} ({BASINS[a // n_block]}): {int(m.sum()):,} CONUS nonzeros; '
                    f'{time.time() - t0:.0f}s'
                )
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    _say(
        f'{vals.size:,} CONUS nonzeros from {np.unique(rows).size:,} rows, pass took {time.time() - t0:.0f}s'
    )

    ev_keep = np.unique(rows)
    mat = sp.csr_matrix(
        (vals, (np.searchsorted(ev_keep, rows), cols)),
        shape=(ev_keep.size, int(keep.sum())),
        dtype=np.float32,
    )
    block = ev_keep // n_block
    idx = ev_keep % n_block
    event_name = [f'ev{i}_{BASINS[b]}' for b, i in zip(block, idx)]
    haz = Hazard(
        haz_type='TC',
        units='m/s',
        centroids=Centroids(lat=xy[keep, 1], lon=xy[keep, 0]),
        event_id=np.arange(1, ev_keep.size + 1),
        frequency=freq[ev_keep],
        event_name=event_name,
        date=date[ev_keep].astype('int64'),
        orig=orig[ev_keep].astype(bool),
        intensity=mat,
    )
    local = ce._local(CROP_KEY)
    haz.write_hdf5(str(local))
    ce._put(s3, local, CROP_KEY)

    nnz_per = np.diff(mat.indptr)
    events = pd.DataFrame(
        {
            'row': ev_keep,
            'basin': [BASINS[b] for b in block],
            'idx': idx,
            'date': date[ev_keep].astype('int64'),
            'frequency': freq[ev_keep],
            'nnz': nnz_per,
            'max_wind': np.asarray(mat.max(axis=1).todense()).ravel(),
        }
    )
    local = ce._local(EVENTS_KEY)
    events.to_parquet(local)
    ce._put(s3, local, EVENTS_KEY)
    for b in BASINS:
        e = events[events.basin == b]
        if len(e):
            _say(f'  {b}: {len(e):,} events with CONUS wind, {int(e.nnz.sum()):,} nonzeros')

    her_cols = np.flatnonzero(keep)
    touched = np.zeros((len(BASINS), her_cols.size), dtype=bool)
    for b in range(len(BASINS)):
        sel = block == b
        if sel.any():
            touched[b, np.unique(mat[sel].indices)] = True
    columns = pd.DataFrame(
        {
            'col': her_cols,
            'lon': xy[keep, 0],
            'lat': xy[keep, 1],
            'cell': colkey[keep],
            'region_id': region_id[keep],
            'on_land': on_land[keep],
            **{f'touched_{b}': touched[i] for i, b in enumerate(BASINS)},
        }
    )
    columns['dup'] = columns.groupby('cell').cumcount()
    local = ce._local(COLS_KEY)
    columns.to_parquet(local)
    ce._put(s3, local, COLS_KEY)
    for i, b in enumerate(BASINS):
        if touched[i].any():
            c = columns[touched[i]]
            _say(
                f'  {b} events touch {len(c):,} columns: lon {c.lon.min():.2f}..{c.lon.max():.2f} '
                f'lat {c.lat.min():.2f}..{c.lat.max():.2f}'
            )

    catalogue = pd.DataFrame({'idx': np.arange(n_block), 'date': date[:n_block].astype('int64')})
    catalogue.attrs = {}
    local = ce._local(CATALOGUE_KEY)
    catalogue.to_parquet(local)
    ce._put(s3, local, CATALOGUE_KEY)

    if not args.no_check:
        check(haz, columns, s3)


def check(haz, columns: pd.DataFrame, s3) -> None:
    """CLIMADA's ranking on the crop against the Dryad points, cell by cell.
    Both sides carry a cell once per basin tile, so values are compared as
    sorted lists per cell."""
    import xarray as xr

    t0 = time.time()
    lv = haz.local_exceedance_intensity(return_periods=ce.RETURN_PERIODS, method=ce.EXTRAPOLATION)[
        0
    ]
    _say(f'ranking on the crop in {time.time() - t0:.0f}s')
    pub = xr.open_dataset(ce._fetch(s3, ce.DRYAD_LEVELS))
    lat, lon = pub['lat'].values, pub['lon'].values
    lo0, la0, lo1, la1 = ce.CONUS_BBOX
    inbox = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)
    pk = cell_key(lon[inbox], lat[inbox])
    out = []
    for T in ce.RETURN_PERIODS:
        mine = pd.DataFrame({'cell': columns.cell.values, 'v': lv[str(T)].values.astype('float64')})
        theirs = pd.DataFrame({'cell': pk, 'v': pub[f'rp_{T}'].values[inbox].astype('float64')})
        mine = mine.sort_values(['cell', 'v']).reset_index(drop=True)
        theirs = theirs.sort_values(['cell', 'v']).reset_index(drop=True)
        n_mine = mine.groupby('cell').size()
        n_theirs = theirs.groupby('cell').size()
        common = n_mine.index.intersection(n_theirs.index)
        same_count = common[(n_mine[common] == n_theirs[common]).values]
        a = mine[mine.cell.isin(same_count)].reset_index(drop=True)
        b = theirs[theirs.cell.isin(same_count)].reset_index(drop=True)
        both = np.isfinite(a.v.values) & np.isfinite(b.v.values)
        d = a.v.values[both] - b.v.values[both]
        ratio = a.v.values[both] / b.v.values[both]
        out.append(
            {
                'rp': T,
                'n': int(both.sum()),
                'cells_count_mismatch': int(len(common) - len(same_count)),
                'within_0.05': float((np.abs(d) < 0.05).mean()),
                'median_abs_diff': float(np.median(np.abs(d))),
                'max_abs_diff': float(np.abs(d).max()),
                'median_ratio': float(np.median(ratio)),
            }
        )
        _say(f'  rp_{T}: {out[-1]}')
    res = pd.DataFrame(out)
    local = ce._local(CHECK_KEY)
    res.to_parquet(local)
    ce._put(s3, local, CHECK_KEY)


# ---------------------------------------------------------------------------
# match: her 400 member runs against the shared track files


def her_runs(s3) -> list[np.ndarray]:
    import h5py

    with (
        s3.open(f'{ce.BUCKET}/{HAZ_KEY}', 'rb', block_size=8 * 2**20) as fo,
        h5py.File(fo, 'r') as h,
    ):
        n = (h['frequency'].shape[0]) // len(BASINS)
        dates = h['date'][:n].astype('int64')
    yrs = np.array([datetime.date.fromordinal(int(x)).year for x in dates])
    cuts = np.r_[0, np.flatnonzero(np.diff(yrs) < 0) + 1, dates.size]
    return [dates[a:b] for a, b in zip(cuts[:-1], cuts[1:])]


def cmd_match(args):
    import xarray as xr

    s3 = ce._s3()
    runs = her_runs(s3)
    _say(f'{len(runs)} member runs in her catalogue; sizes {[r.size for r in runs[:6]]} ...')
    run_sets = [collections.Counter(r.tolist()) for r in runs]
    y0, y1 = ce.parse_range(args.years)[0], ce.parse_range(args.years)[-1]
    found = []
    summary = []
    for i in ce.parse_range(args.files):
        key = f'{ce.TRACKS}/{ce.track_file(i)}'
        local = Path(ce.track_file(i))
        t0 = time.time()
        s3.get(f'{ce.BUCKET}/{key}', str(local))
        ds = xr.open_dataset(local)
        # CLIMADA's CHAZ reader keeps the nodes with a valid time and wind,
        # selects storms by the year of node 0, and dates the event at its
        # first kept node; the same rule here, per member
        t = ds['time'].values
        w = ds['Mwspd'].values  # (member, lifelength, storm)
        ds.close()
        local.unlink()
        tt = pd.to_datetime(t.ravel()).values.reshape(t.shape)
        ok_t = ~pd.isna(tt)
        ord_all = np.full(t.shape, -1, dtype=np.int64)
        ord_all[ok_t] = [pd.Timestamp(x).toordinal() for x in tt[ok_t]]
        year0 = pd.DatetimeIndex(tt[0]).year.values
        inwin = (year0 >= y0) & (year0 <= y1)
        seqs, storms_of = [], []
        for m in range(w.shape[0]):
            valid = ok_t & np.isfinite(w[m])
            has = valid.any(axis=0) & inwin
            first = np.argmax(valid, axis=0)
            storms_of.append(np.flatnonzero(has))
            seqs.append(ord_all[first[has], storms_of[-1]])
        sizes = sorted(s.size for s in seqs)
        size_groups = [
            g
            for g in range(len(runs) // 40)
            if sorted(r.size for r in runs[40 * g : 40 * g + 40]) == sizes
        ]
        fd = collections.Counter(np.concatenate(seqs).tolist())
        cover = max(sum(min(c, fd[d]) for d, c in r.items()) / r.total() for r in run_sets)
        hits = []
        for m, seq in enumerate(seqs):
            for k, r in enumerate(runs):
                if r.size == seq.size and np.array_equal(r, seq):
                    hits.append((k, m))
                    found.append(
                        pd.DataFrame(
                            {
                                'run': k,
                                'pos': np.arange(seq.size),
                                'file': i,
                                'member': m,
                                'storm': storms_of[m],
                            }
                        )
                    )
        summary.append(
            {
                'file': i,
                'storms': int(t.shape[1]),
                'in_window': int(inwin.sum()),
                'member_storms_mean': float(np.mean(sizes)),
                'size_multiset_groups': str(size_groups),
                'exact_hits': len(hits),
                'best_cover': cover,
            }
        )
        _say(
            f'file {i:02d}: {t.shape[1]:,} storms, {int(inwin.sum()):,} in window, {np.mean(sizes):.0f} per member; '
            f'run-size multiset matches her file groups {size_groups}; exact (run, member) hits {hits[:4]}'
            f'{"..." if len(hits) > 4 else ""}; best coverage {cover:.3f}; {time.time() - t0:.0f}s'
        )
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False), flush=True)
    if found:
        mapping = pd.concat(found, ignore_index=True)
        local = ce._local(MATCH_KEY)
        mapping.to_parquet(local)
        ce._put(s3, local, MATCH_KEY)
        _say(f'{mapping.run.nunique()} of {len(runs)} runs mapped to shared track files')
    else:
        _say('no member run of her catalogue is in the shared track files')


# ---------------------------------------------------------------------------
# submit: one Coiled task per subcommand


def cmd_submit(args):
    import coiled

    s3 = ce._s3()
    here = Path(__file__).resolve()
    code = {p.name: f'{ce.HAZARD}/code/{p.name}' for p in (here, here.parent / 'chaz_events.py')}
    for p in (here, here.parent / 'chaz_events.py'):
        ce._put(s3, p, code[p.name])
    fetch = '\n'.join(f"s3fs.S3FileSystem().get('{ce.BUCKET}/{k}', '{n}')" for n, k in code.items())
    script = f"""#!/bin/bash
set -uo pipefail
exec > >(tee task.log) 2>&1
echo "task $COILED_BATCH_TASK_INPUT on $(hostname) $(date -u +%FT%TZ)"
python - <<'PY'
import s3fs
{fetch}
PY
cat > upload_log.py <<'PY'
import os, s3fs
name = 'published_' + os.environ.get('COILED_BATCH_TASK_INPUT', 'task')
s3fs.S3FileSystem().put('task.log', '{ce.BUCKET}/{ce.HAZARD}/logs/' + name + '.log')
PY
python chaz_published.py $COILED_BATCH_TASK_INPUT &
pid=$!
while kill -0 $pid 2>/dev/null; do sleep 60; python upload_log.py || true; done
wait $pid
code=$?
echo "exit $code $(date -u +%FT%TZ)"
python upload_log.py
exit $code
"""
    if args.dry_run:
        print(script)
        return
    result = coiled.batch.run(
        script,
        command_as_script=True,
        name=args.name,
        software=args.software,
        region='us-west-2',
        vm_type=[args.vm_type],
        scheduler_vm_type=[args.vm_type],
        arm=True,
        disk_size=args.disk_size,
        spot_policy='spot_with_fallback',
        max_workers=len(args.tasks),
        forward_aws_credentials=True,
        max_retries=1,
        tag={'Project': 'OCR'},
        map_over_values=args.tasks,
    )
    print(f'submitted: cluster {result.get("cluster_id")}, job {result.get("job_id")}')


# ---------------------------------------------------------------------------
# tailset: the crop in the files chaz_tail_test.py reads


def catalogue_runs(catalogue: pd.DataFrame) -> np.ndarray:
    """Member run of each storm in her catalogue: the storms are in file,
    member, storm order, so the year sequence restarts at every member."""
    yrs = np.array([datetime.date.fromordinal(int(x)).year for x in catalogue['date'].values])
    run = np.cumsum(np.r_[0, np.diff(yrs) < 0])
    return run


def cmd_tailset(args):
    import h5py
    import scipy.sparse as sp
    import xarray as xr
    from climada.hazard import Centroids

    s3 = ce._s3()
    ours = Centroids.from_hdf5(ce._fetch(s3, ce.CENTROIDS_KEY))
    ko = cell_key(ours.lon, ours.lat)
    cols = pd.read_parquet(ce._fetch(s3, COLS_KEY))
    pos = pd.Series(np.arange(len(cols)), index=cell_key(cols['lon'].values, cols['lat'].values))
    sel = pos.reindex(ko).values
    if np.isnan(sel).any():
        raise RuntimeError(f'{int(np.isnan(sel).sum())} CONUS centroids missing from the crop')
    sel = sel.astype(int)

    with h5py.File(ce._fetch(s3, CROP_KEY), 'r') as f:
        g = f['intensity']
        mat = sp.csr_matrix(
            (g['data'][:], g['indices'][:], g['indptr'][:]), shape=tuple(g.attrs['shape'])
        )
        freq = f['frequency'][:]
        names = f['event_name'][:].astype(str)
        date = f['date'][:]
    mat = mat[:, sel].tocsr()
    keep = np.diff(mat.indptr) > 0
    mat, freq, names, date = mat[keep], freq[keep], names[keep], date[keep]
    _say(
        f'{mat.shape[0]:,} events x {mat.shape[1]:,} cells, {mat.nnz:,} nonzeros on the CONUS centroids'
    )

    key = ce.set_key(args.tag)
    local = ce._local(key)
    with h5py.File(local, 'w') as f:
        g = f.create_group('intensity')
        g.attrs['shape'] = mat.shape
        g.create_dataset('data', data=mat.data)
        g.create_dataset('indices', data=mat.indices)
        g.create_dataset('indptr', data=mat.indptr)
        f.create_dataset('frequency', data=freq)
        f.create_dataset('event_name', data=names.astype('S'))
        f.create_dataset('date', data=date)
    ce._put(s3, local, key)

    pub = xr.open_dataset(ce._fetch(s3, ce.DRYAD_LEVELS))
    thr = xr.open_dataset(ce._fetch(s3, ce.DRYAD_THRESH))
    kp = cell_key(pub['lon'].values, pub['lat'].values)
    ip = pd.Series(np.arange(kp.size), index=kp).reindex(ko).values
    if np.isnan(ip).any():
        raise RuntimeError('CONUS centroids missing from the Dryad points')
    ip = ip.astype(int)
    thr_vars = {v: [n for n in thr.data_vars if n.endswith(str(v))] for v in ce.THRESHOLDS}
    data = {
        f'rp_{T}': ('points', pub[f'rp_{T}'].values[ip].astype('float32'))
        for T in ce.RETURN_PERIODS
    }
    for v, names_v in thr_vars.items():
        if len(names_v) == 1:
            data[f'thr_{v:.0f}'] = ('points', thr[names_v[0]].values[ip].astype('float32'))
    levels = xr.Dataset(
        data,
        coords={'lat': ('points', ours.lat), 'lon': ('points', ours.lon)},
        attrs={
            'source': 'chaz_published.py tailset: the Dryad ERA5 points on the CONUS centroids',
            'method': ce.EXTRAPOLATION,
            'files': ','.join(str(i) for i in range(10)),
            'years': '1981-2019',
            'n_events': int(mat.shape[0]),
        },
    )
    key = ce.levels_key(args.tag)
    local = ce._local(key)
    levels.to_netcdf(local)
    ce._put(s3, local, key)

    catalogue = pd.read_parquet(ce._fetch(s3, CATALOGUE_KEY))
    run = catalogue_runs(catalogue)
    n_members = 40
    if run.max() + 1 != 10 * n_members:
        raise RuntimeError(f'{run.max() + 1} member runs in the catalogue, expected 400')
    idx = np.array([int(n.split('_')[0][2:]) for n in names])
    events = pd.DataFrame(
        {
            'event_name': names,
            'basin': [n.split('_')[1] for n in names],
            'idx': idx,
            'file': run[idx] // n_members,
            'member': run[idx] % n_members,
            'year': [datetime.date.fromordinal(int(d)).year for d in date],
            'frequency': freq,
        }
    )
    key = f'{ce.HAZARD}/events/{args.tag}.parquet'
    local = ce._local(key)
    events.to_parquet(local, index=False)
    ce._put(s3, local, key)
    _say(
        f'{len(events):,} events over {events.file.nunique()} files x {events.member.nunique()} members, '
        f'{events.year.min()}-{events.year.max()}'
    )


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest='cmd', required=True)

    q = sub.add_parser('crop', help='CONUS crop of the published hazard, plus the Dryad check')
    q.add_argument(
        '--hazard', default='hazard.hdf5', help='local copy of the global file; fetched if absent'
    )
    q.add_argument('--step', type=int, default=200_000, help='rows per read')
    q.add_argument('--no-check', action='store_true')
    q.set_defaults(func=cmd_crop)

    q = sub.add_parser('match', help='her member runs against the shared track files')
    q.add_argument('--files', default='0-39')
    q.add_argument('--years', default='1981-2019')
    q.set_defaults(func=cmd_match)

    q = sub.add_parser('tailset', help='set, levels and events files for chaz_tail_test.py')
    q.add_argument('--tag', default='published')
    q.set_defaults(func=cmd_tailset)

    q = sub.add_parser('submit', help='run subcommands on Coiled, one VM each')
    q.add_argument('--tasks', default='crop,match', type=lambda s: s.split(','))
    q.add_argument('--software', default='ocr-climada')
    q.add_argument('--vm-type', default='m8g.2xlarge')
    q.add_argument('--disk-size', type=int, default=150, help='GB')
    q.add_argument('--name', default='chaz-published')
    q.add_argument('--dry-run', action='store_true')
    q.set_defaults(func=cmd_submit)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
