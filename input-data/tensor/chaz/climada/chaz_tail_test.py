"""Score the six-level EAD integral against the CHAZ event set.

chaz_events.py leaves a CONUS hazard set on S3: a sparse events-by-cells wind
matrix with a frequency per event, plus the six return levels CLIMADA ranks
out of it. With those, the expected annual damage at unit exposure is exact,

    ead_true = sum_i  f_i * D(v_i)          over the events reaching a cell

and the served product's reconstruction from six levels (chaz_damage.py) can
be compared to it cell by cell, split by the return-period range the damage
comes from, and re-run on subsets and perturbations of the same events. All
of it is numpy on the matrix; CLIMADA is not needed here.

  decompose   truth vs reconstruction, split into T < 10, 10-1000, > 1000 yr
  subsample   many draws at the CMIP6 stores' sample size (8 members x 20 yr)
  perturb     frequency and intensity rescaled the way warming moves them

Return levels for subsets are ranked here with CLIMADA's rule (log-log
interpolation between ranked events, constant beyond the rarest one) and the
implementation is checked against the levels file on the full set first.

  pixi run python input-data/tensor/chaz/climada/chaz_tail_test.py check --tag era5_10f
  pixi run python input-data/tensor/chaz/climada/chaz_tail_test.py decompose --tag era5_10f
  pixi run python input-data/tensor/chaz/climada/chaz_tail_test.py subsample --tag era5_10f --draws 50
  pixi run python input-data/tensor/chaz/climada/chaz_tail_test.py perturb --tag era5_10f

Results land beside the set, under CHAZ/hazard/ERA5/tail/<tag>_[<calibration>_]*, with
no calibration tag for the product's TDR1.0.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chaz_damage import (  # noqa: E402
    CAL_TAG,
    EAD_LOGT_GRID,
    RP_YEARS,
    V_THRESH,
    VHALF,
    WIND_CAP,
    _mdr,
    ead_from_levels,
)

BUCKET = 'carbonplan-ocr'
PREFIX = 'ocr-explore/CHAZ'
HAZARD = f'{PREFIX}/hazard/ERA5'
CACHE = Path(os.environ.get('CHAZ_EVENTS_DIR', Path.home() / '.cache' / 'ocr' / 'chaz-events'))
YRLY_FREQ_IB = {'NA': 10.8, 'EP': 14.5}  # chaz_events.YRLY_FREQ_IB

REGION = 'NA2'
LITPOP_KEY = f'{PREFIX}/validation/litpop_0300as_2020_global.hdf5'
US = 840  # ISO 3166 numeric
RANGES = {'T<10': (0.0, 1.0), '10-1000': (1.0, 3.0), 'T>1000': (3.0, np.inf)}  # log10(T)
VARIANTS = ('slope', 'flat', 'truncate')


def _s3():
    return s3fs.S3FileSystem()


def _local(key: str) -> Path:
    """Where an S3 key lives in the cache; writers use it too, so a fresh
    output is never shadowed by a stale download of the same size."""
    local = CACHE / key.removeprefix(PREFIX + '/')
    local.parent.mkdir(parents=True, exist_ok=True)
    return local


def _stale(s3, local: Path, key: str) -> bool:
    """A local copy is stale when the object on S3 is newer or a different
    size; equal sizes alone say nothing, since same-shaped outputs match."""
    info = s3.info(f'{BUCKET}/{key}')
    return (
        info['size'] != local.stat().st_size
        or info['LastModified'].timestamp() > local.stat().st_mtime
    )


def _fetch(s3, key: str) -> Path:
    local = _local(key)
    if not local.exists() or _stale(s3, local, key):
        s3.get(f'{BUCKET}/{key}', str(local))
    return local


def _put(s3, local: Path, key: str) -> None:
    s3.put(str(local), f'{BUCKET}/{key}')
    os.utime(local)
    print(f'-> s3://{BUCKET}/{key}')


# ---------------------------------------------------------------------------
# the set


class EventSet:
    """Sparse winds (events x cells) with a frequency per event."""

    def __init__(
        self, intensity: sparse.csr_matrix, frequency: np.ndarray, names: np.ndarray, lat, lon
    ):
        self.intensity = intensity.tocsr()
        self.frequency = np.asarray(frequency, dtype='float64')
        self.names = np.asarray(names)
        self.lat, self.lon = np.asarray(lat), np.asarray(lon)

    @property
    def n_cells(self) -> int:
        return self.intensity.shape[1]

    def subset(self, mask: np.ndarray, frequency: np.ndarray | None = None) -> EventSet:
        return EventSet(
            self.intensity[mask],
            self.frequency[mask] if frequency is None else frequency,
            self.names[mask],
            self.lat,
            self.lon,
        )

    def scaled(self, intensity: float = 1.0, frequency: float = 1.0) -> EventSet:
        return EventSet(
            self.intensity * intensity, self.frequency * frequency, self.names, self.lat, self.lon
        )


def load_set(tag: str, s3) -> tuple[EventSet, xr.Dataset]:
    with h5py.File(_fetch(s3, f'{HAZARD}/sets/{tag}.hdf5')) as f:
        g = f['intensity']
        inten = sparse.csr_matrix(
            (g['data'][:], g['indices'][:], g['indptr'][:]), shape=tuple(g.attrs['shape'])
        )
        freq = f['frequency'][:]
        names = f['event_name'][:].astype(str)
    levels = xr.open_dataset(_fetch(s3, f'{HAZARD}/levels/{tag}.nc')).load()
    if inten.shape[1] != levels.sizes['points']:
        raise ValueError('set and levels disagree on the number of cells')
    return EventSet(inten, freq, names, levels['lat'].values, levels['lon'].values), levels


def present_pairs(es: EventSet) -> set[tuple[int, int]]:
    """(file, member) pairs the set holds, from the event names."""
    return {
        tuple(map(int, n.removeprefix('ens').replace('-s', ' ').replace('-m', ' ').split()[::2]))
        for n in es.names
    }


def result_key(tag: str, calibration: str, name: str) -> str:
    return f'{HAZARD}/tail/{tag}_{CAL_TAG[calibration]}{name}'


def load_events(files: list[int], s3) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(_fetch(s3, f'{HAZARD}/events/ens{i:03d}.parquet')) for i in files],
        ignore_index=True,
    )


def cell_key(lat, lon):
    return np.round(lat * 240).astype(np.int64) * 200_000 + np.round(lon * 240).astype(np.int64)


def exposure_weights(es: EventSet, s3) -> np.ndarray:
    """LitPop 2020 value on our cells, US only, zero elsewhere: the weighting
    the walkthrough's event-set comparison uses."""
    with h5py.File(_fetch(s3, LITPOP_KEY), 'r') as f:
        g = f['exposures']
        cols = {c.decode(): g['block1_values'][:, i] for i, c in enumerate(g['block1_items'][:])}
        cols |= {c.decode(): g['block2_values'][:, i] for i, c in enumerate(g['block2_items'][:])}
    us = cols['region_id'] == US
    _, i_c, i_l = np.intersect1d(
        cell_key(es.lat, es.lon),
        cell_key(cols['latitude'][us], cols['longitude'][us]),
        return_indices=True,
    )
    w = np.zeros(es.n_cells)
    w[i_c] = cols['value'][us][i_l]
    return w


def basin_frequency(events: pd.DataFrame) -> pd.Series:
    """Per-event frequency from basin counts, indexed by event name; the same
    rule as chaz_events.basin_weights."""
    w_na = YRLY_FREQ_IB['NA'] / max(int(events['in_na'].sum()), 1)
    w_ep = YRLY_FREQ_IB['EP'] / max(int(events['in_ep'].sum()), 1)
    f = w_na * events['in_na'].values + w_ep * events['in_ep'].values
    return pd.Series(f, index=events['event_name'].values)


# ---------------------------------------------------------------------------
# ranking: per cell, events sorted by wind, cumulative frequency


class Ranked:
    def __init__(self, es: EventSet):
        coo = es.intensity.tocoo()
        keep = coo.data > 0
        r, c, v = coo.row[keep], coo.col[keep], coo.data[keep].astype('float64')
        order = np.lexsort((-v, c))
        self.col, self.val, self.freq = c[order], v[order], es.frequency[r[order]]
        cs = np.cumsum(self.freq)
        self.start = np.searchsorted(self.col, np.arange(es.n_cells), side='left')
        self.end = np.searchsorted(self.col, np.arange(es.n_cells), side='right')
        base = np.concatenate([[0.0], cs])[self.start]
        self.cumfreq = cs - base[self.col]  # exceedance frequency of each entry at its cell
        self.n_cells = es.n_cells
        self._key_scale = 1.0 / (self.cumfreq.max() * 1.01 + 1e-12)
        self._keys = self.col + self.cumfreq * self._key_scale

    def exceedance_intensity(self, return_periods=RP_YEARS) -> np.ndarray:
        """(len(T), n_cells) winds, CLIMADA's local_exceedance_intensity with
        extrapolate_constant: log-log interpolation between ranked events,
        the largest wind beyond the rarest event, zero below the most
        frequent one."""
        out = np.zeros((len(return_periods), self.n_cells))
        cells = np.arange(self.n_cells)
        has = self.end > self.start
        for k, T in enumerate(return_periods):
            f = 1.0 / T
            pos = np.searchsorted(self._keys, cells + f * self._key_scale, side='left')
            at_top = has & (pos <= self.start)
            out[k, at_top] = self.val[self.start[at_top]]
            inside = has & (pos > self.start) & (pos < self.end)
            p = pos[inside]
            f0, f1 = np.log(self.cumfreq[p - 1]), np.log(self.cumfreq[p])
            v0, v1 = np.log(self.val[p - 1]), np.log(self.val[p])
            w = (np.log(f) - f0) / (f1 - f0)
            out[k, inside] = np.exp(v0 + w * (v1 - v0))
        return out

    def ead_by_range(self, v_half) -> dict[str, np.ndarray]:
        """Exact EAD per cell, split by each event's own return period at the cell."""
        dmg = _mdr(self.val, v_half) * self.freq
        logt = -np.log10(self.cumfreq)
        out = {}
        for name, (lo, hi) in RANGES.items():
            m = (logt >= lo) & (logt < hi)
            out[name] = np.bincount(self.col[m], weights=dmg[m], minlength=self.n_cells)
        out['total'] = np.bincount(self.col, weights=dmg, minlength=self.n_cells)
        return out

    def max_return_period(self) -> np.ndarray:
        """Longest resolved return period per cell (the rarest event's)."""
        out = np.full(self.n_cells, np.nan)
        has = self.end > self.start
        out[has] = 1.0 / self.cumfreq[self.start[has]]
        return out


# ---------------------------------------------------------------------------
# reconstruction from six levels, split by range


def wind_on_grid(levels: np.ndarray, variant: str) -> np.ndarray:
    """Wind on EAD_LOGT_GRID from the (6, n) levels, per chaz_damage's model:
    linear in log T between the bands, extended on the end slopes, capped.
    'flat' holds the rp_1000 wind past rp_1000; 'truncate' zeroes it."""
    knots_t = np.log10(RP_YEARS)
    slope = np.maximum(levels[1] - levels[0], 0.0) / (knots_t[1] - knots_t[0])
    knots_t = np.concatenate([[0.0], knots_t])
    knots_v = np.concatenate([(levels[0] - slope)[None], levels])
    seg = np.clip(np.searchsorted(knots_t, EAD_LOGT_GRID, side='right') - 1, 0, len(knots_t) - 2)
    w = (EAD_LOGT_GRID - knots_t[seg]) / (knots_t[seg + 1] - knots_t[seg])
    v = knots_v[seg] * (1.0 - w[:, None]) + knots_v[seg + 1] * w[:, None]
    v = np.minimum(v, WIND_CAP)
    beyond = EAD_LOGT_GRID > knots_t[-1]
    if variant == 'flat':
        v[beyond] = levels[-1]
    elif variant == 'truncate':
        v[beyond] = 0.0
    return v


def recon_by_range(levels: np.ndarray, v_half, variant: str = 'slope') -> dict[str, np.ndarray]:
    """The six-level integral per cell, split by the T range it integrates over.
    The 'slope' total equals chaz_damage.ead_from_levels."""
    d = _mdr(wind_on_grid(levels, variant), v_half)
    lam = 1.0 / 10.0**EAD_LOGT_GRID
    seg_mid = 0.5 * (EAD_LOGT_GRID[1:] + EAD_LOGT_GRID[:-1])
    seg_area = (lam[:-1] - lam[1:])[:, None] * 0.5 * (d[:-1] + d[1:])
    out = {}
    for name, (lo, hi) in RANGES.items():
        m = (seg_mid >= lo) & (seg_mid < hi)
        out[name] = seg_area[m].sum(axis=0)
    out['T>1000'] = out['T>1000'] + lam[-1] * d[-1]  # damage held past the grid
    out['total'] = sum(out[k] for k in RANGES)
    return out


# ---------------------------------------------------------------------------
# reports


def hazard_bins(ref_wind: np.ndarray):
    edges = [V_THRESH, 30, 35, 40, 45, 50, 55, 100]
    for lo, hi in zip(edges, edges[1:]):
        m = (ref_wind >= lo) & (ref_wind < hi)
        if m.sum() >= 20:
            yield f'{lo:.0f}-{hi:.0f}', m


def print_ranges(
    truth: dict, recon: dict[str, dict], mask: np.ndarray, label: str, weights=None
) -> None:
    w = np.ones(mask.size) if weights is None else weights
    tot = (truth['total'] * w)[mask].sum()
    unit = 'yr^-1' if weights is None else '$/yr'
    print(f'\n{label}: {int(mask.sum()):,} cells, true EAD sum {tot:.4g} {unit}')
    print(f'{"":12s} {"truth":>10s} ' + ' '.join(f'{v:>10s}' for v in recon))
    for r in [*RANGES, 'total']:
        line = f'{r:12s} {(truth[r] * w)[mask].sum() / tot:10.4f} '
        line += ' '.join(f'{(recon[v][r] * w)[mask].sum() / tot:10.4f}' for v in recon)
        print(line + ('   (share of true total)' if r == 'T<10' else ''))


def cmd_check(args):
    s3 = _s3()
    es, levels = load_set(args.tag, s3)
    rk = Ranked(es)
    ours = rk.exceedance_intensity()
    print(
        f'{args.tag}: {es.intensity.shape[0]:,} events x {es.n_cells:,} cells, {es.intensity.nnz:,} nonzeros'
    )
    for k, T in enumerate(RP_YEARS):
        ref = levels[f'rp_{int(T)}'].values.astype('float64')
        ok = np.isfinite(ref)
        diff = np.abs(ours[k][ok] - ref[ok])
        print(
            f'  rp_{int(T):<5d} max |ours - climada| = {diff.max():.3e} m/s, cells > 1e-3: {int((diff > 1e-3).sum()):,}'
        )
    tmax = rk.max_return_period()
    print(
        f'  rarest resolved return period: median {np.nanmedian(tmax):,.0f} yr, max {np.nanmax(tmax):,.0f} yr'
    )
    # the slope total must equal the product's integral
    lv = ours
    a = recon_by_range(lv, VHALF[args.calibration][REGION])['total']
    b = ead_from_levels(lv, VHALF[args.calibration][REGION])
    print(f'  recon_by_range total vs ead_from_levels: max diff {np.abs(a - b).max():.3e}')


def cmd_decompose(args):
    s3 = _s3()
    es, levels = load_set(args.tag, s3)
    v_half = VHALF[args.calibration][REGION]
    rk = Ranked(es)
    lv = rk.exceedance_intensity()
    truth = rk.ead_by_range(v_half)
    recon = {v: recon_by_range(lv, v_half, v) for v in VARIANTS}
    live = truth['total'] > 0
    print_ranges(truth, recon, live, f'{args.tag} all cells with damage, unit exposure')
    value = exposure_weights(es, s3)
    print_ranges(truth, recon, live & (value > 0), f'{args.tag} US cells, LitPop-weighted', value)
    for name, m in hazard_bins(lv[3]):
        print_ranges(truth, recon, m & live, f'rp_100 wind {name} m/s, unit exposure')
    ds = xr.Dataset(
        {
            **{f'true_{k}': ('points', truth[k]) for k in truth},
            **{f'{v}_{k}': ('points', recon[v][k]) for v in VARIANTS for k in recon[v]},
            **{f'rp_{int(T)}': ('points', lv[i]) for i, T in enumerate(RP_YEARS)},
            'max_return_period': ('points', rk.max_return_period()),
            'litpop_value_us': ('points', value),
        },
        coords={'lat': ('points', es.lat), 'lon': ('points', es.lon)},
        attrs={
            'tag': args.tag,
            'calibration': args.calibration,
            'v_half': v_half,
            'n_events': es.intensity.shape[0],
        },
    )
    key = result_key(args.tag, args.calibration, 'decompose.nc')
    ds.to_netcdf(_local(key))
    _put(s3, _local(key), key)


def parse_design(s: str) -> dict[str, int]:
    """'files=10,members=8,years=20' -> dict."""
    return {k: int(v) for k, v in (p.split('=') for p in s.split(','))}


def cmd_subsample(args):
    s3 = _s3()
    es, levels = load_set(args.tag, s3)
    v_half = VHALF[args.calibration][REGION]
    files = [int(x) for x in levels.attrs['files'].split(',')]
    y0, y1 = (int(y) for y in levels.attrs['years'].split('-'))
    published = args.tag.startswith('published')
    if published:
        # her catalogue: one row per (storm, basin copy) with her constant
        # basin frequency; a subset's weight is that frequency over the
        # fraction of simulated years drawn, since the basin count scales
        # with the sample
        events = pd.read_parquet(_fetch(s3, f'{HAZARD}/events/{args.tag}.parquet'))
        present = set(zip(events['file'], events['member']))
    else:
        # only the (file, member) pairs the set holds may enter the weights
        present = present_pairs(es)
        events = load_events(files, s3)
        events = events[events['year'].between(y0, y1)]
        events = events[[(f, m) in present for f, m in zip(events['file'], events['member'])]]
    files = sorted({f for f, _ in present})
    members = sorted({m for _, m in present})
    design = parse_design(args.design)
    rng = np.random.default_rng(args.seed)
    full_truth = Ranked(es).ead_by_range(v_half)['total']
    value = exposure_weights(es, s3)
    name_to_row = pd.Series(np.arange(es.intensity.shape[0]), index=es.names)
    rows = []
    for d in range(args.draws):
        f_sel = rng.choice(files, size=min(design['files'], len(files)), replace=False)
        m_sel = rng.choice(members, size=min(design['members'], len(members)), replace=False)
        ys = rng.integers(y0, y1 - design['years'] + 2)
        sub = events[
            events['file'].isin(f_sel)
            & events['member'].isin(m_sel)
            & events['year'].between(ys, ys + design['years'] - 1)
        ]
        if published:
            frac = (
                len(f_sel)
                * len(m_sel)
                * design['years']
                / (len(files) * len(members) * (y1 - y0 + 1))
            )
            freq = pd.Series(sub['frequency'].values / frac, index=sub['event_name'].values)
            reach = sub['event_name'].values
        else:
            freq = basin_frequency(sub)
            reach = sub[sub['reach_conus']]['event_name'].values
        reach = reach[np.isin(reach, es.names)]
        idx = name_to_row.loc[reach].values
        mask = np.zeros(es.intensity.shape[0], bool)
        mask[idx] = True
        ss = es.subset(mask, frequency=freq.loc[es.names[mask]].values)
        rk = Ranked(ss)
        lv = rk.exceedance_intensity()
        truth = rk.ead_by_range(v_half)
        rec = {v: recon_by_range(lv, v_half, v)['total'] for v in VARIANTS}
        live = truth['total'] > 0
        row = {
            'draw': d,
            'n_events': int(mask.sum()),
            'sim_years': len(f_sel) * design['members'] * design['years'],
            'max_rp_median': float(np.nanmedian(rk.max_return_period())),
            'true_sub': float(truth['total'].sum()),
            'true_full': float(full_truth[live].sum()),
            'tail_share_true': float(truth['T>1000'].sum() / truth['total'].sum()),
        }
        for v in VARIANTS:
            row[f'{v}_over_sub'] = float(rec[v].sum() / truth['total'].sum())
            row[f'{v}_over_full'] = float(rec[v].sum() / full_truth.sum())
            row[f'{v}_over_sub_$'] = float((rec[v] * value).sum() / (truth['total'] * value).sum())
            row[f'{v}_over_full_$'] = float((rec[v] * value).sum() / (full_truth * value).sum())
        rows.append(row)
        print(
            f'draw {d:3d}: {row["n_events"]:7,d} events, {row["sim_years"]:,} sim-yr, '
            f'median rarest T {row["max_rp_median"]:,.0f}; recon/sub-truth '
            + ' '.join(f'{v} {row[f"{v}_over_sub"]:.3f}' for v in VARIANTS)
            + f'; slope/full-truth {row["slope_over_full"]:.3f}'
        )
    df = pd.DataFrame(rows)
    print(
        f'\n{args.tag} design {args.design}, {args.draws} draws (ratios of aggregate EAD at unit exposure)'
    )
    print(f'{"":24s} {"median":>8s} {"IQR":>17s}')
    for c in (
        [f'{v}_over_sub' for v in VARIANTS]
        + [f'{v}_over_full' for v in VARIANTS]
        + [f'{v}_over_sub_$' for v in VARIANTS]
        + [f'{v}_over_full_$' for v in VARIANTS]
        + ['tail_share_true']
    ):
        q = df[c].quantile([0.25, 0.5, 0.75]).values
        print(f'{c:24s} {q[1]:8.3f}   {q[0]:7.3f}-{q[2]:7.3f}')
    design = args.design.replace('=', '').replace(',', '_')
    df['calibration'] = args.calibration
    key = result_key(args.tag, args.calibration, f'subsample_{design}.parquet')
    df.to_parquet(_local(key), index=False)
    _put(s3, _local(key), key)


def cmd_perturb(args):
    s3 = _s3()
    es, levels = load_set(args.tag, s3)
    v_half = VHALF[args.calibration][REGION]
    value = exposure_weights(es, s3)
    cases = [('base', 1.0, 1.0)]
    cases += [(f'freq x{f}', 1.0, f) for f in (0.7, 1.3)]
    cases += [(f'wind x{f}', f, 1.0) for f in (1.05, 1.10)]
    cases += [('wind x1.05, freq x0.7', 1.05, 0.7), ('wind x1.10, freq x1.3', 1.10, 1.3)]
    print(f'{args.tag}: aggregate reconstruction / truth under rescaled events')
    print(
        f'{"case":24s} {"true EAD":>10s} {"tail share":>10s} '
        + ' '.join(f'{v:>9s}' for v in VARIANTS)
        + '  | LitPop-weighted: '
        + ' '.join(f'{v:>9s}' for v in VARIANTS)
    )
    rows = []
    for name, fi, ff in cases:
        rk = Ranked(es.scaled(intensity=fi, frequency=ff))
        lv = rk.exceedance_intensity()
        truth = rk.ead_by_range(v_half)
        tot = truth['total'].sum()
        tot_v = (truth['total'] * value).sum()
        recon = {v: recon_by_range(lv, v_half, v)['total'] for v in VARIANTS}
        rec = {v: recon[v].sum() / tot for v in VARIANTS}
        rec_v = {f'{v}_$': (recon[v] * value).sum() / tot_v for v in VARIANTS}
        rows.append(
            {
                'case': name,
                'true_total': tot,
                'tail_share': truth['T>1000'].sum() / tot,
                **rec,
                **rec_v,
            }
        )
        print(
            f'{name:24s} {tot:10.4g} {truth["T>1000"].sum() / tot:10.3f} '
            + ' '.join(f'{rec[v]:9.3f}' for v in VARIANTS)
            + '  | '
            + ' '.join(f'{rec_v[f"{v}_$"]:9.3f}' for v in VARIANTS)
        )
    df = pd.DataFrame(rows)
    df['calibration'] = args.calibration
    key = result_key(args.tag, args.calibration, 'perturb.parquet')
    df.to_parquet(_local(key), index=False)
    _put(s3, _local(key), key)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--tag', required=True)
    p.add_argument('--calibration', default='TDR1.0', choices=list(VHALF))
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check', help='ranking against the CLIMADA levels file').set_defaults(
        func=cmd_check
    )
    sub.add_parser('decompose', help='truth vs reconstruction by return-period range').set_defaults(
        func=cmd_decompose
    )
    q = sub.add_parser('subsample', help='draws at a smaller sample size')
    q.add_argument(
        '--design', default='files=10,members=8,years=20', help='the CMIP6 stores: 10 x 8 x 20'
    )
    q.add_argument('--draws', type=int, default=50)
    q.add_argument('--seed', type=int, default=0)
    q.set_defaults(func=cmd_subsample)
    sub.add_parser('perturb', help='rescaled frequency and intensity').set_defaults(
        func=cmd_perturb
    )
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
