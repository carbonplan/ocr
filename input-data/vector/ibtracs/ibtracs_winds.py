"""Model the wind each IBTrACS storm since 1980 produced at every CONUS cell.

Historic tracks alone can't say whether a storm mattered to a location: the
USA wind radii only start in 2001, and a track 150 km away can still bring
hurricane-force winds. So each storm is run through the same Holland (2008)
wind model, on the same 300 arcsec CONUS centroids, as the CHAZ hazard maps
(see input-data/tensor/chaz/climada/chaz_events.py), and the peak 1-min
sustained wind per storm per cell is packed onto the served CHAZ grid.

  winds   climada env: IBTrACS (USA agency, NA + EP basins) -> CLIMADA
          TropCyclone -> ~/.cache/ocr/ibtracs/winds/{intensity.npz,storms.csv}
  store   default env: scatter onto the 1/12 degree CHAZ grid ->
          s3://carbonplan-ocr/ocr-explore/ibtracs_conus_winds.zarr
          wind(storm, lat, lon) uint8 m/s, 0 = below 17.5 m/s; storm table
          in the group attrs and as storms.json beside the store

  E=~/dev/carbonplan/ocr/.pixi/envs
  $E/climada/bin/python ibtracs_winds.py winds
  $E/default/bin/python ibtracs_winds.py store [--force]
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

BUCKET = 'carbonplan-ocr'
PREFIX = 'ocr-explore'
CENTROIDS_KEY = f'{PREFIX}/CHAZ/hazard/centroids_conus_0300as.hdf5'
GRID_STORE = f'{PREFIX}/CHAZ/processed/chaz_conus_v2_ERA5_points/0'
KEY = f'{PREFIX}/ibtracs_conus_winds.zarr'
STORMS_KEY = f'{PREFIX}/ibtracs_conus_winds/storms.json'
ARTIFACT_ID = 'ibtracs_conus_winds'
IBTRACS_FILE = 'IBTrACS.since1980.v04r01.nc'
YEAR_RANGE = (1980, datetime.date.today().year)
BASINS = ('NA', 'EP')

CONUS_BBOX = (-125.0, 24.0, -66.0, 50.5)
BUFFER_DEG = 5.0
INTENSITY_THRES = 17.5  # m/s, CLIMADA's wind-field floor (34 kt)
KT_PER_MS = 1 / 0.514444
TIME_STEP_H = 0.5
WIND_MODEL = 'H08'
CROP_MARGIN = 2
CHUNK = 16

CACHE = Path.home() / '.cache' / 'ocr' / 'ibtracs' / 'winds'
INTENSITY = CACHE / 'intensity.npz'
STORMS = CACHE / 'storms.csv'
CENTROIDS = CACHE / 'centroids_conus_0300as.hdf5'

log = logging.getLogger('ibtracs_winds')


def _s3():
    import s3fs

    return s3fs.S3FileSystem()


def in_conus_reach(lat, lon) -> np.ndarray:
    lo0, la0, lo1, la1 = CONUS_BBOX
    return (
        (lon >= lo0 - BUFFER_DEG)
        & (lon <= lo1 + BUFFER_DEG)
        & (lat >= la0 - BUFFER_DEG)
        & (lat <= la1 + BUFFER_DEG)
    )


# ---------------------------------------------------------------------------
# winds (climada env)


def crop_to_reach(track):
    hit = np.nonzero(in_conus_reach(track['lat'].values, track['lon'].values))[0]
    lo = max(int(hit[0]) - CROP_MARGIN, 0)
    hi = min(int(hit[-1]) + CROP_MARGIN + 1, track.sizes['time'])
    return track.isel(time=slice(lo, hi))


def reaches_conus(track) -> bool:
    strong = track['max_sustained_wind'].values >= INTENSITY_THRES * KT_PER_MS
    return bool((strong & in_conus_reach(track['lat'].values, track['lon'].values)).any())


_CENT = None


def _init_worker(path: str):
    global _CENT
    from climada.hazard import Centroids

    logging.getLogger('climada').setLevel(logging.WARNING)
    _CENT = Centroids.from_hdf5(path)


def _winds(data):
    from climada.hazard import TCTracks, TropCyclone

    haz = TropCyclone.from_tracks(TCTracks(data=data), centroids=_CENT, model=WIND_MODEL)
    haz.event_name = [tr.sid for tr in data]
    return haz


def load_tracks():
    from climada.hazard import TCTracks

    seen = set()
    data = []
    for basin in BASINS:
        tracks = TCTracks.from_ibtracs_netcdf(
            provider='usa',
            year_range=YEAR_RANGE,
            basin=basin,
            estimate_missing=True,
            file_name=IBTRACS_FILE,
        )
        for tr in tracks.data:
            if tr.sid in seen or not reaches_conus(tr):
                continue
            seen.add(tr.sid)
            tr.attrs['basin_query'] = basin
            data.append(tr)
        log.info('%s: %d tracks read, %d reaching CONUS so far', basin, tracks.size, len(data))
    return TCTracks(data=data)


def storm_record(tr) -> dict:
    t = tr['time'].values
    return {
        'sid': tr.sid,
        'name': tr.name,
        'season': int(tr.attrs.get('season', np.datetime64(t[0], 'Y').astype(int) + 1970)),
        'basin': tr.attrs['basin_query'],
        'start': str(np.datetime_as_string(t[0], unit='D')),
        'end': str(np.datetime_as_string(t[-1], unit='D')),
        'vmax': round(float(np.nanmax(tr['max_sustained_wind'].values)) / KT_PER_MS, 1),
    }


def cmd_winds(args) -> None:
    import pandas as pd
    from climada.hazard import Hazard
    from scipy import sparse

    CACHE.mkdir(parents=True, exist_ok=True)
    s3 = _s3()
    if not CENTROIDS.exists():
        s3.get(f'{BUCKET}/{CENTROIDS_KEY}', str(CENTROIDS))
    t0 = time.time()
    tracks = load_tracks()
    records = [storm_record(tr) for tr in tracks.data]
    tracks.data = [crop_to_reach(tr) for tr in tracks.data]
    tracks.equal_timestep(TIME_STEP_H)
    log.info('%d tracks prepared in %.0fs', tracks.size, time.time() - t0)

    t1 = time.time()
    workers = args.workers
    parts = [tracks.data[k::workers] for k in range(workers)]
    with mp.get_context('spawn').Pool(
        workers, initializer=_init_worker, initargs=(str(CENTROIDS),)
    ) as pool:
        hazards = list(pool.imap_unordered(_winds, [p for p in parts if p]))
    haz = Hazard.concat(hazards)
    order = {sid: i for i, sid in enumerate(haz.event_name)}
    rows = np.array([order[r['sid']] for r in records])
    intensity = sparse.csr_matrix(haz.intensity)[rows]
    dt = time.time() - t1
    print(
        f'{tracks.size} tracks, {intensity.getnnz():,} nonzeros, {dt:.0f}s '
        f'({dt / tracks.size:.2f} s/track), max {intensity.max():.1f} m/s'
    )
    sparse.save_npz(INTENSITY, intensity)
    np.savez(
        CACHE / 'centroids.npz',
        lat=haz.centroids.lat,
        lon=haz.centroids.lon,
    )
    pd.DataFrame(records).to_csv(STORMS, index=False)
    print(f'wrote {INTENSITY} and {STORMS}')


# ---------------------------------------------------------------------------
# store (default env)


def grid_coords() -> tuple[np.ndarray, np.ndarray]:
    import zarr

    store = zarr.storage.FsspecStore.from_url(f's3://{BUCKET}/{GRID_STORE}', read_only=True)
    group = zarr.open_group(store, mode='r')
    return group['lat'][:], group['lon'][:]


def nearest_index(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    step = abs(coords[1] - coords[0])
    idx = np.abs(coords[None, :] - values[:, None]).argmin(axis=1)
    off = np.abs(coords[idx] - values)
    if off.max() > step / 2:
        raise RuntimeError(f'centroids off the grid by up to {off.max():.4f} deg')
    return idx


def cmd_store(args) -> None:
    import pandas as pd
    import zarr
    from scipy import sparse

    sys.path.insert(0, str(Path(__file__).parent))
    from ibtracs_tracks import _write_manifest

    s3 = _s3()
    if s3.exists(f'{BUCKET}/{KEY}/zarr.json') and not args.force:
        sys.exit(f's3://{BUCKET}/{KEY} exists; use --force to overwrite')
    intensity = sparse.load_npz(INTENSITY).tocsr()
    cent = np.load(CACHE / 'centroids.npz')
    storms = pd.read_csv(STORMS)
    assert len(storms) == intensity.shape[0]

    hit = np.asarray(intensity.max(axis=1).todense()).ravel() > 0
    storms = storms[hit].reset_index(drop=True)
    intensity = intensity[hit]
    order = np.lexsort((storms['start'].values, storms['season'].values))
    storms = storms.iloc[order].reset_index(drop=True)
    intensity = intensity[order]
    print(f'{len(storms)} storms with winds >= {INTENSITY_THRES} m/s on CONUS cells')

    lat, lon = grid_coords()
    iy = nearest_index(lat, cent['lat'])
    ix = nearest_index(lon, cent['lon'])
    wind = np.zeros((len(storms), lat.size, lon.size), dtype='uint8')
    coo = intensity.tocoo()
    wind[coo.row, iy[coo.col], ix[coo.col]] = np.clip(np.rint(coo.data), 1, 255)

    records = storms.to_dict(orient='records')
    local = CACHE / KEY.rsplit('/', 1)[-1]
    root = zarr.create_group(zarr.storage.LocalStore(local), zarr_format=3, overwrite=True)
    root.attrs.update(
        {
            'storms': records,
            'units': 'm/s',
            'description': (
                'Peak 1-min sustained wind per storm per cell, Holland (2008) model on '
                'IBTrACS v04r01 USA-agency tracks (CLIMADA); 0 below 17.5 m/s'
            ),
            'attribution': 'IBTrACS v04r01, NOAA NCEI; wind model via CLIMADA',
        }
    )
    root.create_array('lat', data=lat, chunks=lat.shape)
    root.create_array('lon', data=lon, chunks=lon.shape)
    arr = root.create_array(
        'wind',
        shape=wind.shape,
        dtype='uint8',
        chunks=(len(storms), CHUNK, CHUNK),
        dimension_names=['storm', 'lat', 'lon'],
        compressors=[zarr.codecs.GzipCodec(level=9)],
    )
    arr[:] = wind
    if s3.exists(f'{BUCKET}/{KEY}'):
        s3.rm(f'{BUCKET}/{KEY}', recursive=True)
    s3.put(str(local), f'{BUCKET}/{KEY}', recursive=True)
    with s3.open(f'{BUCKET}/{STORMS_KEY}', 'w') as fh:
        json.dump(records, fh)
    size = sum(f.stat().st_size for f in local.rglob('*') if f.is_file()) / 1e6
    print(f'uploaded -> s3://{BUCKET}/{KEY} ({size:.1f} MB)')

    _write_manifest(
        s3,
        {
            'id': ARTIFACT_ID,
            'format': 'zarr',
            'url': f'https://{BUCKET}.s3.us-west-2.amazonaws.com/{KEY}',
            'storms_url': f'https://{BUCKET}.s3.us-west-2.amazonaws.com/{STORMS_KEY}',
            'arrays': {'wind': ['storm', 'lat', 'lon']},
            'units': 'm/s',
            'storm_count': len(storms),
            'grid': GRID_STORE,
            'chunks': [len(storms), CHUNK, CHUNK],
            'source': {'file': IBTRACS_FILE, 'provider': 'usa', 'basins': list(BASINS)},
            'transform': (
                'IBTrACS since-1980 tracks with tropical-storm winds within 5 degrees of '
                'CONUS, missing pressure/RMW estimated by CLIMADA, resampled to 30 min, '
                f'Holland 2008 winds ({WIND_MODEL}) on the CHAZ 300 arcsec centroids, '
                'peak per cell rounded to whole m/s and scattered onto the served grid'
            ),
            'attribution': 'IBTrACS v04r01, NOAA NCEI',
            'built': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'),
        },
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('winds', help='run the CLIMADA wind model (climada env)')
    w.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('store', help='pack the winds onto the CHAZ grid and upload')
    s.add_argument('--force', action='store_true')
    args = p.parse_args()
    {'winds': cmd_winds, 'store': cmd_store}[args.cmd](args)


if __name__ == '__main__':
    main()
