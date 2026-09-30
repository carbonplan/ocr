"""Build a CONUS wind-hazard event set from the CHAZ ERA5 tracks with CLIMADA.

The hazard maps served from CHAZ/processed are six return levels per cell.
chaz_damage.py integrates those into an expected annual damage, and past the
1000-year band the integral extrapolates. The only way to score that
extrapolation is against the full event set: with every synthetic storm's wind
at every cell, the damage-frequency curve is exact and the six-level
reconstruction can be compared to it cell by cell. This module builds that
event set for CONUS the way the published maps were built (Meiler et al. 2026;
github.com/simonameiler/CHAZ-hazard-maps):

  tracks    CLIMADA's CHAZ reader, tracks resampled to a 0.5 h step
  winds     Holland (2008) parametric model, 1-min sustained 10-m wind,
            CLIMADA defaults otherwise (17.5 m/s floor, 300 km eye radius)
  cells     the 300 arcsec land points of the published ERA5 maps inside the
            CONUS box, with the published centroids' distance to coast attached
  weights   one frequency per basin, the IBTrACS annual count divided by the
            number of simulated events with tropical-storm winds in that basin
  levels    empirical exceedance ranking with constant extrapolation, the
            published maps' method

Only storms that carry tropical-storm winds within 5 degrees of the CONUS box
are computed; the rest of the globe is never touched. Basin membership is
tested on the raw tracks against simple North Atlantic / East Pacific
outlines rather than on the hazard, which is what the published weights count.
Storms crossing the Central American divide count in both basins, as they do
in the published concatenation.

Steps, each writing under s3://carbonplan-ocr/ocr-explore/CHAZ/hazard/ERA5/:

  centroids                         centroids_conus_0300as.hdf5
  events    --files 0-39            events/ens###.parquet, subsets/ens###.nc
  hazard    --file 3 --members 0-7  h08/ens003_m00-07.hdf5   (one Coiled task)
  batch     --files 0-9             submits the hazard tasks to Coiled
  levels    --files 0-9 --years 1981-2019 --tag era5_10f
                                    sets/<tag>.hdf5, levels/<tag>.nc
  check     --tag era5_10f          levels/<tag>_check.png, ratio table

Run in the `climada` pixi environment (pixi run -e climada python ...).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

BUCKET = 'carbonplan-ocr'
PREFIX = 'ocr-explore/CHAZ'
TRACKS = f'{PREFIX}/tracks/ERA5'
HAZARD = f'{PREFIX}/hazard/ERA5'
CENTROIDS_KEY = f'{PREFIX}/hazard/centroids_conus_0300as.hdf5'
DRYAD_LEVELS = (
    f'{PREFIX}/exceedance_intensity/nc/ERA5/TC_global_0300as_CHAZ_ERA5_exceedance_intensity.nc'
)
DRYAD_THRESH = f'{PREFIX}/return_periods/nc/ERA5/TC_global_0300as_CHAZ_ERA5_return_periods.nc'

CACHE = Path(os.environ.get('CHAZ_EVENTS_DIR', Path.home() / '.cache' / 'ocr' / 'chaz-events'))

CONUS_BBOX = (
    -125.0,
    24.0,
    -66.0,
    50.5,
)  # lon_min, lat_min, lon_max, lat_max; chaz_damage.CONUS_BBOX
BUFFER_DEG = 5.0  # tracks this far outside the box can still reach it (300 km eye radius)
INTENSITY_THRES = 17.5  # m/s, CLIMADA's wind-field floor
KT_PER_MS = 1 / 0.514444
TIME_STEP_H = 0.5
TRACK_BATCH = 50  # tracks per from_tracks call
TIME_FILL = -54786.0  # days since 1950 in the raw files, marking no data
WIND_MODEL = 'H08'
N_FILES = 40
N_MEMBERS = 40

RETURN_PERIODS = (10, 25, 50, 100, 250, 1000)
THRESHOLDS = (33.0, 50.0)
EXTRAPOLATION = 'extrapolate_constant'

# IBTrACS annual counts the published maps correct to (freq_corr_era5.py upstream)
YRLY_FREQ_IB = {'NA': 10.8, 'EP': 14.5}

log = logging.getLogger('chaz_events')


def track_file(i: int) -> str:
    return f'global_2019_2ens{i:03d}_pre.nc'


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out.extend(range(int(a), int(b or a) + 1))
    return out


def event_name(file_i: int, storm: int, member: int) -> str:
    return f'ens{file_i:03d}-s{storm}-m{member}'


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


def _fetch(s3, key: str, force: bool = False) -> Path:
    """Copy an S3 object into the local cache, once."""
    local = _local(key)
    if force or not local.exists() or _stale(s3, local, key):
        s3.get(f'{BUCKET}/{key}', str(local))
    return local


def _put(s3, local: Path, key: str) -> None:
    s3.put(str(local), f'{BUCKET}/{key}')
    os.utime(local)
    log.info('-> s3://%s/%s', BUCKET, key)


# ---------------------------------------------------------------------------
# basins


def wrap_lon(lon):
    return np.where(lon > 180, lon - 360, lon)


def in_north_atlantic(lat, lon) -> np.ndarray:
    """Atlantic side of the Central American divide, 5-60N."""
    lon = wrap_lon(lon)
    atl = (lat > 5) & (lat < 60) & (lon > -100) & (lon < 0)
    pacific = ((lon < -84) & (lat < 9)) | ((lon < -90) & (lat < 15)) | ((lon < -98) & (lat < 18))
    return atl & ~pacific


def in_east_pacific(lat, lon) -> np.ndarray:
    lon = wrap_lon(lon)
    pac = (lat > 5) & (lat < 60) & (lon < -75) & (lon > -180)
    return pac & ~in_north_atlantic(lat, lon)


def in_conus_reach(lat, lon) -> np.ndarray:
    lon = wrap_lon(lon)
    lo0, la0, lo1, la1 = CONUS_BBOX
    return (
        (lon >= lo0 - BUFFER_DEG)
        & (lon <= lo1 + BUFFER_DEG)
        & (lat >= la0 - BUFFER_DEG)
        & (lat <= la1 + BUFFER_DEG)
    )


# ---------------------------------------------------------------------------
# centroids


def cmd_centroids(args):
    from climada.hazard import Centroids
    from climada.util.api_client import Client
    from scipy.spatial import cKDTree

    s3 = _s3()
    if not args.force and s3.exists(f'{BUCKET}/{CENTROIDS_KEY}'):
        print(f'exists: s3://{BUCKET}/{CENTROIDS_KEY} (use --force)')
        return
    pts = xr.open_dataset(_fetch(s3, DRYAD_LEVELS))
    lat, lon = pts['lat'].values, pts['lon'].values
    lo0, la0, lo1, la1 = CONUS_BBOX
    inbox = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)
    cent = Centroids(lat=lat[inbox], lon=lon[inbox])

    # CLIMADA 6 reads distance to coast only from a NASA raster that is no
    # longer hosted; the API's 150 arcsec centroids carry the same distances,
    # and the 300 arcsec lattice is a subset of theirs.
    api = Client().get_centroids(
        res_arcsec_land=150, res_arcsec_ocean=1800, extent=(-180, 180, -60, 60)
    )
    g = api.gdf.cx[lo0 - 0.1 : lo1 + 0.1, la0 - 0.1 : la1 + 0.1]
    d, j = cKDTree(np.c_[g.geometry.x.values, g.geometry.y.values]).query(
        np.c_[cent.gdf.geometry.x.values, cent.gdf.geometry.y.values]
    )
    if d.max() * 3600 > 1.0:
        raise RuntimeError(
            f'Dryad cells are not on the API lattice (max offset {d.max() * 3600:.1f} arcsec)'
        )
    cent.gdf['dist_coast'] = g['dist_coast'].values[j]
    cent.gdf['on_land'] = g['on_land'].values[j]
    far = int((cent.gdf['dist_coast'].values > 1000e3).sum())
    print(f'{cent.size:,} CONUS centroids; {far:,} more than 1000 km inland (zeroed by CLIMADA)')
    local = _local(CENTROIDS_KEY)
    cent.write_hdf5(str(local))
    _put(s3, local, CENTROIDS_KEY)


def load_centroids(s3):
    from climada.hazard import Centroids

    return Centroids.from_hdf5(str(_fetch(s3, CENTROIDS_KEY)))


# ---------------------------------------------------------------------------
# events: which (member, storm) pairs matter, and which basins they belong to


def events_key(i: int) -> str:
    return f'{HAZARD}/events/ens{i:03d}.parquet'


def subset_key(i: int) -> str:
    return f'{HAZARD}/subsets/ens{i:03d}.nc'


def build_events(i: int, s3, force: bool) -> pd.DataFrame:
    if (
        not force
        and s3.exists(f'{BUCKET}/{events_key(i)}')
        and s3.exists(f'{BUCKET}/{subset_key(i)}')
    ):
        return pd.read_parquet(_fetch(s3, events_key(i)))
    src = _fetch(s3, f'{TRACKS}/{track_file(i)}')
    # raw time encoding is kept so the subset decodes exactly like the source
    ds = xr.open_dataset(src, engine='netcdf4', decode_times=False)
    lat, lon = ds['latitude'].values, ds['longitude'].values  # (lifelength, storm)
    w = ds['Mwspd'].values  # (member, lifelength, storm), kt
    year = ds['year'].values
    strong = w >= INTENSITY_THRES * KT_PER_MS
    reach = (strong & in_conus_reach(lat, lon)[None]).any(axis=1)  # (member, storm)
    na = (strong & in_north_atlantic(lat, lon)[None]).any(axis=1)
    ep = (strong & in_east_pacific(lat, lon)[None]).any(axis=1)
    vmax = np.nanmax(np.where(np.isfinite(w), w, -np.inf), axis=1)
    n_m, n_s = reach.shape
    member, storm = np.meshgrid(np.arange(n_m), np.arange(n_s), indexing='ij')
    df = pd.DataFrame(
        {
            'file': np.full(reach.size, i, dtype='int16'),
            'storm': storm.ravel().astype('int32'),
            'member': member.ravel().astype('int16'),
            'year': np.broadcast_to(year, reach.shape).ravel().astype('int16'),
            'vmax_kt': vmax.ravel().astype('float32'),
            'reach_conus': reach.ravel(),
            'in_na': na.ravel(),
            'in_ep': ep.ravel(),
        }
    )
    df['event_name'] = [event_name(i, s, m) for s, m in zip(df['storm'], df['member'])]
    df = df[df['in_na'] | df['in_ep'] | df['reach_conus']].reset_index(drop=True)

    keep = np.nonzero(reach.any(axis=0))[0]
    # nodes past a storm's end carry the fill date, which decodes to a real
    # date; CLIMADA drops nodes on wind, so blank the wind there instead
    fill = ds['time'].values <= TIME_FILL
    ds['Mwspd'] = ds['Mwspd'].where(~fill[None])
    sub_local = _local(subset_key(i))
    ds.isel(stormID=keep).to_netcdf(sub_local)
    ev_local = _local(events_key(i))
    df.to_parquet(ev_local, index=False)
    _put(s3, sub_local, subset_key(i))
    _put(s3, ev_local, events_key(i))
    print(
        f'ens{i:03d}: {n_s:,} storms; {len(keep):,} reach CONUS in some member; '
        f'{int(reach.sum()):,} reaching events; NA {int(na.sum()):,}, EP {int(ep.sum()):,}'
    )
    return df


def load_events(files: list[int], s3) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(_fetch(s3, events_key(i))) for i in files], ignore_index=True)


def cmd_events(args):
    s3 = _s3()
    for i in parse_range(args.files):
        build_events(i, s3, args.force)


# ---------------------------------------------------------------------------
# hazard: wind fields for one file and a run of intensity members


def hazard_key(i: int, m0: int, m1: int) -> str:
    return f'{HAZARD}/h08/ens{i:03d}_m{m0:02d}-{m1:02d}.hdf5'


def parse_task(task: str) -> tuple[int, int, int]:
    """'3:0-7' -> (3, 0, 7)."""
    f, _, m = task.partition(':')
    m0, _, m1 = m.partition('-')
    return int(f), int(m0), int(m1 or m0)


_CENT = None


def _init_worker(centroids_path: str):
    global _CENT
    from climada.hazard import Centroids

    logging.getLogger('climada').setLevel(logging.WARNING)
    _CENT = Centroids.from_hdf5(centroids_path)


def _winds(payload):
    from climada.hazard import TCTracks, TropCyclone

    data, max_memory_gb = payload
    # from_tracks keeps a per-track hazard, each with its own copy of the
    # centroids, until it concatenates: ~12 MB per track. Small batches
    # bound that.
    parts = []
    for k in range(0, len(data), TRACK_BATCH):
        batch = data[k : k + TRACK_BATCH]
        haz = TropCyclone.from_tracks(
            TCTracks(data=batch), centroids=_CENT, model=WIND_MODEL, max_memory_gb=max_memory_gb
        )
        haz.event_name = [tr.sid for tr in batch]
        parts.append(haz)
    return parts[0] if len(parts) == 1 else TropCyclone.concat(parts)


def winds_parallel(tracks, cent, workers: int, max_memory_gb: float):
    """Holland winds for all tracks, split across processes. Each worker
    loads the centroids once; CLIMADA's own pool ships them per call."""
    import multiprocessing as mp

    from climada.hazard import Hazard

    if workers <= 1 or tracks.size < 2 * workers:
        _init_worker(str(_local(CENTROIDS_KEY)))
        return _winds((tracks.data, max_memory_gb))
    parts = [tracks.data[k::workers] for k in range(workers)]
    ctx = mp.get_context('spawn')
    hazards = []
    t0 = time.time()
    with ctx.Pool(
        workers, initializer=_init_worker, initargs=(str(_local(CENTROIDS_KEY)),)
    ) as pool:
        # CLIMADA's budget is per call, so each worker gets its share
        for haz in pool.imap_unordered(
            _winds, [(p_, max_memory_gb / workers) for p_ in parts if p_]
        ):
            hazards.append(haz)
            log.info('part %d/%d done, %.0fs', len(hazards), len(parts), time.time() - t0)
    return Hazard.concat(hazards)


CROP_MARGIN = 2  # nodes kept on each side of the reaching segment


def crop_to_reach(track):
    """Keep the span of nodes within reach of the CONUS box, plus a margin.
    Winds reach 300 km from the eye, so nodes outside the 5-degree buffer
    contribute nothing to CONUS cells, and CLIMADA's memory and time scale
    with the number of nodes."""
    hit = np.nonzero(in_conus_reach(track['lat'].values, track['lon'].values))[0]
    lo = max(int(hit[0]) - CROP_MARGIN, 0)
    hi = min(int(hit[-1]) + CROP_MARGIN + 1, track.sizes['time'])
    return track.isel(time=slice(lo, hi))


def compute_hazard(i: int, m0: int, m1: int, s3, force: bool, max_memory_gb: float, workers: int):
    from climada.hazard import TCTracks

    key = hazard_key(i, m0, m1)
    if not force and s3.exists(f'{BUCKET}/{key}'):
        print(f'exists: s3://{BUCKET}/{key}')
        return
    t0 = time.time()
    cent = load_centroids(s3)
    events = pd.read_parquet(_fetch(s3, events_key(i)))
    reaching = events[events['reach_conus']]
    wanted = reaching[reaching['member'].between(m0, m1)]
    sub = _fetch(s3, subset_key(i))
    tracks = TCTracks.from_simulations_chaz(str(sub), ensemble_nums=list(range(m0, m1 + 1)))
    # the reader names tracks <file>-<storm>-<member>, with storm numbered by
    # position in the subset file (the kept storms in ascending order)
    kept_storms = np.sort(reaching['storm'].unique())
    pos = np.searchsorted(kept_storms, wanted['storm'].values)
    rename = {
        f'{sub.name}-{p_}-{m}': n for p_, m, n in zip(pos, wanted['member'], wanted['event_name'])
    }
    kept = []
    for tr in tracks.data:
        if tr.sid in rename:
            tr.attrs['sid'] = tr.attrs['name'] = rename[tr.sid]
            kept.append(tr)
    tracks.data = [crop_to_reach(tr) for tr in kept]
    log.info(
        'ens%03d m%02d-%02d: %d reaching tracks read in %.0fs',
        i,
        m0,
        m1,
        tracks.size,
        time.time() - t0,
    )
    tracks.equal_timestep(TIME_STEP_H)
    t1 = time.time()
    haz = winds_parallel(tracks, cent, workers, max_memory_gb)
    dt = time.time() - t1
    print(
        f'ens{i:03d} m{m0:02d}-{m1:02d}: {tracks.size:,} tracks, {haz.intensity.getnnz():,} nonzeros, '
        f'{dt:.0f}s ({dt / max(tracks.size, 1):.2f} s/track), max {haz.intensity.max():.1f} m/s'
    )
    local = _local(key)
    haz.write_hdf5(str(local))
    _put(s3, local, key)


def cmd_hazard(args):
    task = args.task or os.environ.get('COILED_BATCH_TASK_INPUT')
    if task:
        i, m0, m1 = parse_task(task)
    else:
        i = args.file
        m0, m1 = parse_range(args.members)[0], parse_range(args.members)[-1]
    compute_hazard(i, m0, m1, _s3(), args.force, args.max_memory_gb, args.workers or os.cpu_count())


def member_groups(per_task: int) -> list[tuple[int, int]]:
    return [(m, min(m + per_task, N_MEMBERS) - 1) for m in range(0, N_MEMBERS, per_task)]


def cmd_batch(args):
    s3 = _s3()
    tasks = []
    for i in parse_range(args.files):
        for m0, m1 in member_groups(args.members_per_task):
            if args.force or not s3.exists(f'{BUCKET}/{hazard_key(i, m0, m1)}'):
                tasks.append(f'{i}:{m0}-{m1}')
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print('nothing to do')
        return
    print(f'{len(tasks)} tasks, e.g. {tasks[0]} .. {tasks[-1]}')
    # the VM fetches the script from our bucket; Coiled's own uploader needs a
    # bucket its role cannot create in this account. The command goes up as a
    # script so no shell quoting survives the trip.
    import coiled

    code_key = f'{HAZARD}/code/{Path(__file__).name}'
    _put(s3, Path(__file__).resolve(), code_key)
    # each task keeps its own log in the bucket; Coiled's task logs are not
    # always retrievable
    force = ' --force' if args.force else ''
    script = f"""#!/bin/bash
set -uo pipefail
exec > >(tee task.log) 2>&1
echo "task $COILED_BATCH_TASK_INPUT on $(hostname) $(date -u +%FT%TZ)"
python - <<'PY'
import s3fs
s3fs.S3FileSystem().get('{BUCKET}/{code_key}', 'chaz_events.py')
PY
cat > upload_log.py <<'PY'
import os, s3fs
name = os.environ.get('COILED_BATCH_TASK_INPUT', 'task').replace(':', '_')
s3fs.S3FileSystem().put('task.log', '{BUCKET}/{HAZARD}/logs/' + name + '.log')
PY
python chaz_events.py hazard --max-memory-gb {args.max_memory_gb} --workers {args.workers}{force} &
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
        spot_policy='spot_with_fallback',
        max_workers=args.max_workers,
        forward_aws_credentials=True,
        max_retries=2,
        tag={'Project': 'OCR'},
        map_over_values=tasks,
    )
    print(f'submitted: cluster {result.get("cluster_id")}, job {result.get("job_id")}')


# ---------------------------------------------------------------------------
# levels: join the chunks, weight the events, rank the winds


def set_key(tag: str) -> str:
    return f'{HAZARD}/sets/{tag}.hdf5'


def levels_key(tag: str) -> str:
    return f'{HAZARD}/levels/{tag}.nc'


def basin_weights(events: pd.DataFrame) -> dict[str, float]:
    """One frequency per basin: the observed annual count over the number of
    simulated events with tropical-storm winds in the basin. The simulated
    year count cancels, so this is insensitive to the sample size."""
    return {
        'NA': YRLY_FREQ_IB['NA'] / int(events['in_na'].sum()),
        'EP': YRLY_FREQ_IB['EP'] / int(events['in_ep'].sum()),
    }


def load_set(files: list[int], years: tuple[int, int], s3):
    from climada.hazard import Hazard

    chunks = []
    for i in files:
        keys = sorted(
            k for k in s3.ls(f'{BUCKET}/{HAZARD}/h08/') if Path(k).name.startswith(f'ens{i:03d}_')
        )
        if not keys:
            raise FileNotFoundError(f'no hazard chunks for ens{i:03d}')
        for k in keys:
            chunks.append(Hazard.from_hdf5(str(_fetch(s3, k.removeprefix(BUCKET + '/')))))
    haz = Hazard.concat(chunks)
    haz.event_id = np.arange(1, haz.size + 1)
    names = np.array(haz.event_name)
    # overlapping chunk ranges (e.g. one chunking scheme replaced by another)
    # would count an event twice; keep the first copy
    _, first = np.unique(names, return_index=True)
    if first.size < names.size:
        log.warning('%d duplicate events across chunks dropped', names.size - first.size)
        haz = haz.select(event_id=list(haz.event_id[np.sort(first)]))
        names = np.array(haz.event_name)

    # basin counts over the (file, member) pairs the chunks actually hold,
    # so a partial member range still gets the right per-event weight
    present = {
        tuple(map(int, n.removeprefix('ens').replace('-s', ' ').replace('-m', ' ').split()[::2]))
        for n in names
    }
    events = load_events(files, s3)
    events = events[events['year'].between(*years)]
    events = events[[(f, m) in present for f, m in zip(events['file'], events['member'])]]
    w = basin_weights(events)
    ev = events[events['reach_conus']].set_index('event_name')
    ev = ev.assign(frequency=w['NA'] * ev['in_na'].values + w['EP'] * ev['in_ep'].values)
    keep = np.isin(names, ev.index.values)
    haz = haz.select(event_names=list(names[keep]))
    haz.frequency = ev.loc[haz.event_name, 'frequency'].values
    haz.event_id = np.arange(1, haz.size + 1)
    n_zero = int((haz.frequency == 0).sum())
    if n_zero:
        log.warning('%d events reach CONUS but fall in neither basin outline; weight 0', n_zero)
    info = {
        'files': files,
        'years': years,
        'n_events': int(haz.size),
        'n_na': int(events['in_na'].sum()),
        'n_ep': int(events['in_ep'].sum()),
        'w_na': w['NA'],
        'w_ep': w['EP'],
    }
    return haz, info


def cmd_levels(args):
    s3 = _s3()
    files = parse_range(args.files)
    years = tuple(int(y) for y in args.years.split('-'))
    t0 = time.time()
    haz, info = load_set(files, years, s3)
    print(
        f'{info["n_events"]:,} events over {len(files)} files, {years[0]}-{years[1]}; '
        f'w_NA = 1/{1 / info["w_na"]:,.0f} yr (N={info["n_na"]:,}), w_EP = 1/{1 / info["w_ep"]:,.0f} yr '
        f'(N={info["n_ep"]:,}); loaded in {time.time() - t0:.0f}s'
    )
    local_set = _local(set_key(args.tag))
    haz.write_hdf5(str(local_set))
    _put(s3, local_set, set_key(args.tag))

    t0 = time.time()
    lv = haz.local_exceedance_intensity(return_periods=RETURN_PERIODS, method=EXTRAPOLATION)[0]
    rp = haz.local_return_period(threshold_intensities=THRESHOLDS, method=EXTRAPOLATION)[0]
    print(f'levels in {time.time() - t0:.0f}s')
    out = xr.Dataset(
        {
            **{f'rp_{T}': ('points', lv[str(T)].values.astype('float32')) for T in RETURN_PERIODS},
            **{
                f'thr_{int(v)}': ('points', rp[str(float(v))].values.astype('float32'))
                for v in THRESHOLDS
            },
        },
        coords={'lat': ('points', haz.centroids.lat), 'lon': ('points', haz.centroids.lon)},
        attrs={
            'source': 'chaz_events.py levels',
            'wind_model': WIND_MODEL,
            'time_step_h': TIME_STEP_H,
            'method': EXTRAPOLATION,
            'files': ','.join(map(str, files)),
            'years': f'{years[0]}-{years[1]}',
            **{k: v for k, v in info.items() if k not in ('files', 'years')},
        },
    )
    local = _local(levels_key(args.tag))
    out.to_netcdf(local)
    _put(s3, local, levels_key(args.tag))


# ---------------------------------------------------------------------------
# check: our levels against the published ERA5 points on the same cells


def cell_key(lat, lon):
    return np.round(lat * 240).astype(np.int64) * 200_000 + np.round(lon * 240).astype(np.int64)


def cmd_check(args):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    s3 = _s3()
    ours = xr.open_dataset(_fetch(s3, levels_key(args.tag)))
    pub = xr.open_dataset(_fetch(s3, DRYAD_LEVELS))
    thr = xr.open_dataset(_fetch(s3, DRYAD_THRESH))
    _, i_o, i_p = np.intersect1d(
        cell_key(ours.lat.values, ours.lon.values),
        cell_key(pub.lat.values, pub.lon.values),
        return_indices=True,
    )
    _, i_o2, i_t = np.intersect1d(
        cell_key(ours.lat.values, ours.lon.values),
        cell_key(thr.lat.values, thr.lon.values),
        return_indices=True,
    )
    print(
        f'{args.tag}: {ours.attrs.get("n_events")} events; {len(i_o):,} shared cells with the published ERA5 points'
    )

    ref = pub['rp_100'].values[i_p]
    edges = [INTENSITY_THRES, 25, 30, 35, 40, 45, 50, 60, 100]
    print('\nmedian ours/published by band, binned by published rp_100 wind (m/s)')
    print(
        f'{"bin":>10s} {"n":>7s} '
        + ' '.join(f'{f"rp_{T}":>8s}' for T in RETURN_PERIODS)
        + '   thr_33   thr_50'
    )
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        m = (ref >= lo) & (ref < hi)
        if m.sum() < 20:
            continue
        r = []
        for T in RETURN_PERIODS:
            a, b = ours[f'rp_{T}'].values[i_o][m], pub[f'rp_{T}'].values[i_p][m]
            ok = (a > 0) & (b > 0)
            r.append(np.median(a[ok] / b[ok]) if ok.sum() else np.nan)
        rt = []
        for v in THRESHOLDS:
            a = ours[f'thr_{int(v)}'].values[i_o2]
            b = thr[f'thr_{int(v)}'].values[i_t]
            mm = np.isin(i_o2, i_o[m]) & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
            rt.append(np.median(a[mm] / b[mm]) if mm.sum() else np.nan)
        rows.append((lo, hi, int(m.sum()), r, rt))
        print(
            f'{lo:>4.0f}-{hi:<5.0f} {m.sum():7,d} '
            + ' '.join(f'{x:8.3f}' for x in r)
            + ' '
            + ' '.join(f'{x:8.3f}' for x in rt)
        )

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    a, b = ours['rp_100'].values[i_o], pub['rp_100'].values[i_p]
    ok = (a > 0) & (b > 0)
    axs[0].hexbin(b[ok], a[ok], gridsize=50, bins='log', cmap='Blues', linewidths=0)
    lim = (INTENSITY_THRES, max(a.max(), b.max()) * 1.02)
    axs[0].plot(lim, lim, color='0.4', lw=0.8, ls=':')
    axs[0].set(
        xlabel='published rp_100 (m/s)',
        ylabel='ours rp_100 (m/s)',
        xlim=lim,
        ylim=lim,
        title='100-yr wind, shared cells',
    )
    for T, c in zip(RETURN_PERIODS, plt.cm.viridis(np.linspace(0, 0.9, len(RETURN_PERIODS)))):
        axs[1].plot(
            [0.5 * (lo + hi) for lo, hi, *_ in rows],
            [r[RETURN_PERIODS.index(T)] for *_, r, _ in rows],
            marker='o',
            color=c,
            label=f'rp_{T}',
        )
    axs[1].axhline(1, color='0.4', lw=0.8, ls=':')
    axs[1].set(
        xlabel='published rp_100 wind (m/s)',
        ylabel='median ours / published',
        title='ratio by hazard bin',
    )
    axs[1].legend(fontsize=8)
    local = _local(f'{HAZARD}/levels/{args.tag}_check.png')
    fig.savefig(local, dpi=130)
    _put(s3, local, f'{HAZARD}/levels/{args.tag}_check.png')
    print(f'figure: {local}')


# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    logging.getLogger('climada').setLevel(logging.WARNING)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    q = sub.add_parser('centroids', help='CONUS centroids from the published ERA5 points')
    q.add_argument('--force', action='store_true')
    q.set_defaults(func=cmd_centroids)

    q = sub.add_parser('events', help='per-file event tables and CONUS track subsets')
    q.add_argument('--files', default=f'0-{N_FILES - 1}')
    q.add_argument('--force', action='store_true')
    q.set_defaults(func=cmd_events)

    q = sub.add_parser('hazard', help='wind fields for one file and member range')
    q.add_argument('--file', type=int)
    q.add_argument('--members', default=f'0-{N_MEMBERS - 1}')
    q.add_argument('--task', help="'<file>:<m0>-<m1>'; defaults to $COILED_BATCH_TASK_INPUT")
    q.add_argument('--max-memory-gb', type=float, default=8.0)
    q.add_argument(
        '--workers', type=int, default=0, help='processes for the wind fields; 0 = all cores'
    )
    q.add_argument('--force', action='store_true')
    q.set_defaults(func=cmd_hazard)

    q = sub.add_parser('batch', help='submit hazard tasks to Coiled')
    q.add_argument('--files', default='0-9')
    q.add_argument('--members-per-task', type=int, default=4)
    q.add_argument('--software', default='ocr-climada')
    q.add_argument('--vm-type', default='m8g.xlarge')
    q.add_argument('--name', default='chaz-hazard')
    q.add_argument('--max-workers', type=int, default=25, help='concurrent VMs')
    q.add_argument('--workers', type=int, default=0, help='processes per task; 0 = all cores')
    q.add_argument('--limit', type=int, default=0, help='submit only the first N pending tasks')
    q.add_argument('--max-memory-gb', type=float, default=8.0, help='for the whole VM')
    q.add_argument('--force', action='store_true')
    q.add_argument('--dry-run', action='store_true')
    q.set_defaults(func=cmd_batch)

    q = sub.add_parser('levels', help='join chunks, weight events, compute return levels')
    q.add_argument('--files', default='0-9')
    q.add_argument('--years', default='1981-2019')
    q.add_argument('--tag', required=True)
    q.set_defaults(func=cmd_levels)

    q = sub.add_parser('check', help='compare a levels file with the published ERA5 points')
    q.add_argument('--tag', required=True)
    q.set_defaults(func=cmd_check)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
