# /// script
# requires-python = ">=3.11"
# dependencies = ["geopandas", "matplotlib", "numpy", "pandas", "pyarrow", "pyproj",
#                 "rasterio", "requests", "scikit-learn", "s3fs", "zarr>=3"]
# ///
"""Local, descriptive temporal validation of the frozen Collins FDP raster.

This is a watershed-matched presence/background ranking evaluation, not a
reproduction of the paper's land-cover/floodplain-constrained pseudo-absences.
No model is fitted; all compute and output writes are local.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from sklearn.metrics import average_precision_score, roc_auc_score

NOAA = 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'
TIFF = 'https://carbonplan-ocr.s3.us-west-2.amazonaws.com/ocr-explore/FDP/raw/CONUS_FDP_100m.tif'
HUC = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/1/query'
BEGIN = pd.Timestamp('2020-06-01')
END = pd.Timestamp('2026-01-01')
SEED = 20261015


def download(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    with requests.get(url, stream=True, timeout=(20, 180)) as r:
        r.raise_for_status()
        with part.open('wb') as out:
            for chunk in r.iter_content(2**20):
                out.write(chunk)
    part.rename(dest)
    print(f'Downloaded {dest.name}: {dest.stat().st_size / 1e6:.1f} MB', flush=True)


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inputs(cache):
    raw = cache / 'noaa'
    raw.mkdir(parents=True, exist_ok=True)
    index = raw / 'index.html'
    download(NOAA, index)
    names = sorted(
        set(
            re.findall(
                r'StormEvents_(?:details|locations)-ftp_v1\.0_d\d{4}_c\d+\.csv\.gz',
                index.read_text(),
            )
        )
    )
    selected = {}
    for name in names:
        kind, year = re.search(r'StormEvents_(\w+)-ftp_v1.0_d(\d{4})', name).groups()
        if (kind == 'details' and 2006 <= int(year) <= 2025) or (
            kind == 'locations' and 2020 <= int(year) <= 2025
        ):
            selected[kind, year] = name  # lexicographically latest creation date
    assert len(selected) == 26, f'Expected 20 details + 6 location files, got {len(selected)}'
    sources = [(NOAA + n, raw / n) for n in selected.values()]
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda pair: download(*pair), sources))
    download(TIFF, cache / 'CONUS_FDP_100m.tif')
    huc_path = cache / 'huc2.geojson'
    if not huc_path.exists():
        params = dict(
            where="huc2 >= '01' AND huc2 <= '18'",
            outFields='huc2,name',
            outSR=5070,
            returnGeometry='true',
            maxAllowableOffset=100,
            f='geojson',
        )
        r = requests.get(HUC, params=params, timeout=90)
        r.raise_for_status()
        assert len(r.json()['features']) == 18
        huc_path.write_bytes(r.content)
        (cache / 'huc2-source-url.txt').write_text(r.url)
    sources += [
        (TIFF, cache / 'CONUS_FDP_100m.tif'),
        ((cache / 'huc2-source-url.txt').read_text(), huc_path),
        (NOAA, index),
    ]
    return selected, [
        dict(url=u, path=str(p), bytes=p.stat().st_size, sha256=sha256(p)) for u, p in sources
    ]


def dollars(series):
    parts = (
        series.fillna('')
        .astype(str)
        .str.strip()
        .str.upper()
        .str.extract(r'^([0-9]+(?:\.[0-9]+)?)\s*([HKMB]?)$')
    )
    return pd.to_numeric(parts[0], errors='coerce') * parts[1].map(
        {'': 1, 'H': 100, 'K': 1000, 'M': 1e6, 'B': 1e9}
    )


def load_reports(cache, selected):
    frames = []
    for (kind, year), name in selected.items():
        if kind != 'details':
            continue
        df = pd.read_csv(cache / 'noaa' / name, low_memory=False)
        df = df[df.EVENT_TYPE.isin(['Flood', 'Flash Flood'])].copy()
        df['date'] = pd.to_datetime(df.BEGIN_DATE_TIME, format='%d-%b-%y %H:%M:%S', errors='raise')
        frames.append(df)
    events = pd.concat(frames, ignore_index=True)
    assert events.EVENT_ID.is_unique, 'Duplicate NOAA EVENT_ID across selected details files'
    events = events[(events.date >= '2006-12-01') & (events.date < END)].copy()
    events['property_usd'] = dollars(events.DAMAGE_PROPERTY)
    events['crop_usd'] = dollars(events.DAMAGE_CROPS)
    casualty = events[
        ['INJURIES_DIRECT', 'INJURIES_INDIRECT', 'DEATHS_DIRECT', 'DEATHS_INDIRECT']
    ].sum(axis=1)
    events['recorded_impact'] = (events.property_usd > 0) | (events.crop_usd > 0) | (casualty > 0)
    events['year'] = events.date.dt.year
    # Exclude AK, HI and territories by administrative state, before spatial QA.
    events['conus_state'] = (events.STATE_FIPS < 60) & ~events.STATE_FIPS.isin([2, 15])
    begin = events.rename(columns={'BEGIN_LAT': 'lat', 'BEGIN_LON': 'lon'}).copy()
    begin['location_policy'] = 'begin'
    locations = pd.concat(
        [
            pd.read_csv(cache / 'noaa' / name)
            for (kind, year), name in selected.items()
            if kind == 'locations'
        ],
        ignore_index=True,
    )
    assert not locations.duplicated(['EVENT_ID', 'LOCATION_INDEX']).any()
    extra = (
        locations[['EVENT_ID', 'LOCATION_INDEX', 'LATITUDE', 'LONGITUDE']]
        .merge(
            events.drop(columns=['BEGIN_LAT', 'BEGIN_LON']), on='EVENT_ID', validate='many_to_one'
        )
        .rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
    )
    extra['location_policy'] = 'locations'
    extra = extra[extra.date >= BEGIN].copy()
    return events, begin, extra


def index_xy(x, y, transform, shape):
    row = np.floor((np.asarray(y) - transform.f) / transform.e).astype('int64')
    col = np.floor((np.asarray(x) - transform.c) / transform.a).astype('int64')
    valid = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    return row, col, valid


def assign_huc(frame, hucs):
    geo = gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(frame.x, frame.y), crs=5070)
    joined = gpd.sjoin(geo, hucs[['huc2', 'geometry']], how='left', predicate='within')
    # Independently simplified neighboring polygons can overlap in narrow strips.
    # Exclude ambiguous points instead of duplicating them or choosing a HUC.
    ambiguous = joined.index[joined.index.duplicated()].unique()
    joined = joined[~joined.index.duplicated()].copy()
    joined.loc[ambiguous, 'huc2'] = None
    joined['ambiguous_huc'] = joined.index.isin(ambiguous)
    return pd.DataFrame(joined.drop(columns=['geometry', 'index_right']))


def score_points(frame, grid, transform, hucs):
    frame = frame.copy()
    frame['qa_status'] = 'ok'
    frame.loc[~frame.conus_state, 'qa_status'] = 'non_conus_state'
    valid_ll = frame.lat.between(24, 50) & frame.lon.between(-126, -66)
    frame.loc[(frame.qa_status == 'ok') & ~valid_ll, 'qa_status'] = (
        'invalid_or_outside_conus_coordinates'
    )
    okay = frame.qa_status == 'ok'
    transformer = Transformer.from_crs(4326, 5070, always_xy=True)
    frame.loc[okay, 'x'], frame.loc[okay, 'y'] = transformer.transform(
        frame.loc[okay, 'lon'].values, frame.loc[okay, 'lat'].values
    )
    valid = frame.loc[okay].copy()
    row, col, inside = index_xy(valid.x, valid.y, transform, grid.shape)
    valid['row'], valid['col'] = row, col
    frame.loc[valid.index[~inside], 'qa_status'] = 'outside_raster_extent'
    valid = valid[inside].copy()
    valid['fdp'] = grid[valid.row.values, valid.col.values]
    valid['cell_id'] = valid.row * grid.shape[1] + valid.col
    finite = np.isfinite(valid.fdp) & valid.fdp.between(0, 1)
    frame.loc[valid.index[~finite], 'qa_status'] = 'raster_nodata'
    valid = assign_huc(valid[finite], hucs)
    frame.loc[valid.index[valid.huc2.isna()], 'qa_status'] = 'outside_current_huc_boundaries'
    frame.loc[valid.index[valid.ambiguous_huc], 'qa_status'] = 'ambiguous_huc_overlap'
    return valid.dropna(subset=['huc2']).copy(), frame


def sample_background(grid, transform, hucs, excluded, target):
    rng = np.random.default_rng(SEED)
    parts = []
    totals = {h: 0 for h in target}
    for iteration in range(100):
        row = rng.integers(0, grid.shape[0], 400_000)
        col = rng.integers(0, grid.shape[1], 400_000)
        score = grid[row, col]
        ids = row * grid.shape[1] + col
        keep = np.isfinite(score) & (score >= 0) & (score <= 1) & ~np.isin(ids, excluded)
        df = pd.DataFrame(dict(row=row[keep], col=col[keep], fdp=score[keep], cell_id=ids[keep]))
        df['x'] = transform.c + (df.col + 0.5) * transform.a
        df['y'] = transform.f + (df.row + 0.5) * transform.e
        df = assign_huc(df, hucs).dropna(subset=['huc2'])
        parts.append(df)
        pool = pd.concat(parts, ignore_index=True).drop_duplicates('cell_id')
        totals = pool.huc2.value_counts().to_dict()
        if all(totals.get(h, 0) >= n for h, n in target.items()):
            result = pd.concat(
                [pool[pool.huc2 == h].head(n) for h, n in target.items()], ignore_index=True
            )
            print(
                f'Background: {len(result):,} unique cells in {iteration + 1} batches', flush=True
            )
            return result
    raise RuntimeError(f'Background sampling failed: {totals}, target={target}')


def top_area_recall(positives, background, fraction):
    """Expected recall with uniform random tie-breaking at the area cutoff.

    Some HUCs have more than 20% of background cells at score 1. Including all
    ties would incorrectly label a much larger area as the top 10% or 20%.
    """
    captured = 0.0
    for huc, sample in positives.groupby('huc2'):
        scores = background.loc[background.huc2 == huc, 'fdp'].to_numpy()
        threshold = np.quantile(scores, 1 - fraction)
        above = np.mean(scores > threshold)
        tied = np.mean(scores == threshold)
        tie_fraction = np.clip((fraction - above) / tied, 0, 1) if tied else 0
        captured += (sample.fdp > threshold).sum() + tie_fraction * (sample.fdp == threshold).sum()
    return float(captured / len(positives))


def balanced_metrics(positives, background):
    # Equal total weight for presences/background and matched HUC distribution.
    counts = positives.huc2.value_counts()
    background = background[background.huc2.isin(counts.index)].copy()
    background['weight'] = background.huc2.map(counts / background.huc2.value_counts())
    joined = pd.concat(
        [positives.assign(label=1, weight=1.0), background.assign(label=0)], ignore_index=True
    )
    auc = roc_auc_score(joined.label, joined.fdp, sample_weight=joined.weight)
    ap = average_precision_score(joined.label, joined.fdp, sample_weight=joined.weight)
    return dict(
        n_positive=len(positives),
        n_background=len(background),
        auc=float(auc),
        average_precision_balanced=float(ap),
        mean_positive=float(positives.fdp.mean()),
        recall_top10_huc_area=top_area_recall(positives, background, 0.1),
        recall_top20_huc_area=top_area_recall(positives, background, 0.2),
    ), joined


def block_bootstrap(joined, n=200):
    # Poisson cluster bootstrap: all points in a 50 km square share a weight.
    block = list(
        zip(np.floor(joined.x / 50_000).astype(int), np.floor(joined.y / 50_000).astype(int))
    )
    ids, unique = pd.factorize(pd.Series(block))
    rng = np.random.default_rng(SEED + 1)
    y, s, w = joined.label.values, joined.fdp.values, joined.weight.values
    values = []
    for _ in range(n):
        weights = w * rng.poisson(1, len(unique))[ids]
        if weights[y == 0].sum() and weights[y == 1].sum():
            values.append(roc_auc_score(y, s, sample_weight=weights))
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def evaluate(points, background, output):
    future = points[points.date >= BEGIN].copy()
    historical = set(points.loc[points.date < BEGIN, 'cell_id'])
    future['seen_in_historical_begin_reports'] = future.cell_id.isin(historical)
    cells = future.sort_values(['date', 'EVENT_ID']).drop_duplicates('cell_id').copy()
    impact_cells = future[future.recorded_impact].sort_values('date').drop_duplicates('cell_id')
    scenarios = {
        'all_reports_unique_cells': cells,
        'recorded_impact_unique_cells': impact_cells,
        'all_reports_event_weighted': future,
        'new_cells_vs_historical_begin': cells[~cells.seen_in_historical_begin_reports],
        'repeated_cells_vs_historical_begin': cells[cells.seen_in_historical_begin_reports],
    }
    summary = {}
    for name, sample in scenarios.items():
        if len(sample):
            summary[name], joined = balanced_metrics(sample, background)
            if name in ['all_reports_unique_cells', 'recorded_impact_unique_cells']:
                summary[name]['auc_95ci_50km_block_poisson'] = block_bootstrap(joined)
    rows = []
    for name, grouping in [('year', 'year'), ('huc2', 'huc2'), ('event_type', 'EVENT_TYPE')]:
        for key, sample in future.groupby(grouping):
            sample = sample.drop_duplicates('cell_id')
            metrics, _ = balanced_metrics(sample, background)
            rows.append(dict(grouping=name, group=str(key), **metrics))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'metrics_by_group.csv', index=False)
    summary['equal_huc_mean_auc'] = float(table[table.grouping == 'huc2'].auc.mean())
    future.to_parquet(output / 'scored_begin_reports.parquet', index=False)
    cells.to_parquet(output / 'scored_unique_cells.parquet', index=False)
    return summary, table, cells


def figures(cells, background, table, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    counts = cells.huc2.value_counts()
    weights = background.huc2.map(counts / background.huc2.value_counts()).fillna(0)
    axes[0].hist(
        background.fdp,
        bins=30,
        weights=weights,
        density=True,
        alpha=0.5,
        label='Watershed-matched background',
        color='#8196aa',
    )
    axes[0].hist(
        cells.fdp,
        bins=30,
        density=True,
        alpha=0.6,
        label='Later reported flood cells',
        color='#008876',
    )
    axes[0].set(
        xlabel='Published FDP score', ylabel='Density', title='Reported locations score higher?'
    )
    axes[0].legend(fontsize=8)
    years = table[table.grouping == 'year']
    axes[1].plot(years.group, years.auc, 'o-', color='#008876')
    axes[1].axhline(0.5, color='gray', linestyle=':')
    axes[1].set(
        ylim=(0.45, 1), ylabel='Presence/background ROC AUC', title='By year (2020 starts June)'
    )
    hucs = table[table.grouping == 'huc2'].sort_values('auc')
    axes[2].barh(hucs.group, hucs.auc, color='#008876')
    axes[2].axvline(0.5, color='gray', linestyle=':')
    axes[2].set(
        xlim=(0, 1),
        xlabel='Presence/background ROC AUC',
        ylabel='HUC-2',
        title='Regional variation',
    )
    fig.suptitle('Frozen Collins FDP surface vs NOAA reports, June 2020–December 2025', fontsize=14)
    fig.savefig(output / 'validation.png', dpi=180)
    plt.close(fig)


def verify_served(grid, points):
    import s3fs
    import zarr
    from zarr.core.sync import sync

    fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
    store = zarr.storage.FsspecStore(
        fs=fs, path='carbonplan-ocr/ocr-explore/FDP/processed/flood_damage_probability/0'
    )
    group = zarr.open_group(store, mode='r')
    sample = points.sample(n=min(12, len(points)), random_state=SEED)
    served = np.array([float(group['fdp'][int(r), int(c)]) for r, c in zip(sample.row, sample.col)])
    sync(fs._s3.__aexit__(None, None, None))
    source = grid[sample.row.values, sample.col.values]
    assert np.allclose(source, served, atol=0, rtol=0), 'Published TIFF and served level-0 differ'
    return dict(n_cells=len(sample), max_absolute_difference=float(np.max(np.abs(source - served))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/ocr/fdp-validation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    selected, manifest = inputs(args.cache)
    print('Reading raster into memory (about 6 GB)', flush=True)
    with rasterio.open(args.cache / 'CONUS_FDP_100m.tif') as src:
        assert src.crs.to_epsg() == 5070
        assert src.shape == (31327, 48357)
        assert src.transform.a == 100 and src.transform.e == -100
        grid = src.read(1)
        if src.nodata is not None:
            grid[grid == src.nodata] = np.nan
        transform = src.transform
    hucs = gpd.read_file(args.cache / 'huc2.geojson').to_crs(5070)
    assert len(hucs) == 18
    events, begin, extra = load_reports(args.cache, selected)
    points, qa = score_points(begin, grid, transform, hucs)
    extra_points, extra_qa = score_points(extra, grid, transform, hucs)
    qa.groupby(['year', 'qa_status']).size().rename('n').to_csv(args.output / 'qa_begin_counts.csv')
    extra_qa.groupby(['year', 'qa_status']).size().rename('n').to_csv(
        args.output / 'qa_location_counts.csv'
    )
    qa[qa.qa_status != 'ok'].to_parquet(args.output / 'excluded_begin_reports.parquet', index=False)
    counts = points[points.date >= BEGIN].drop_duplicates('cell_id').huc2.value_counts()
    excluded = np.union1d(points.cell_id, extra_points.cell_id)
    target = {h: max(10_000, int(n) * 5) for h, n in counts.items()}
    print('Sampling same-watershed background', flush=True)
    background = sample_background(grid, transform, hucs, excluded, target)
    background.to_parquet(args.output / 'background.parquet', index=False)
    print('Computing metrics and spatial-block uncertainty', flush=True)
    summary, table, cells = evaluate(points, background, args.output)
    extra_points.to_parquet(args.output / 'scored_location_reports.parquet', index=False)
    for label, subset in [
        ('all_location_rows_unique_cells', extra_points),
        ('all_location_rows_impact_unique_cells', extra_points[extra_points.recorded_impact]),
    ]:
        summary[label], _ = balanced_metrics(subset.drop_duplicates('cell_id'), background)
    summary['source_vs_served_check'] = verify_served(grid, points[points.date >= BEGIN])
    summary['raw_events_in_test_period'] = int((events.date >= BEGIN).sum())
    summary['raw_conus_events_in_test_period'] = int(
        ((events.date >= BEGIN) & events.conus_state).sum()
    )
    summary['historical_begin_unique_cells'] = int(points[points.date < BEGIN].cell_id.nunique())
    summary['valid_test_events'] = int((points.date >= BEGIN).sum())
    figures(cells, background, table, args.output)
    summary['elapsed_seconds'] = time.time() - start
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest = dict(
        created_utc=datetime.now(UTC).isoformat(),
        seed=SEED,
        period=['2020-06-01', '2025-12-31'],
        sources=manifest,
        script_sha256=sha256(Path(__file__)),
        packages={
            name: importlib.metadata.version(name)
            for name in [
                'geopandas',
                'matplotlib',
                'numpy',
                'pandas',
                'pyarrow',
                'pyproj',
                'rasterio',
                'requests',
                'scikit-learn',
                's3fs',
                'zarr',
            ]
        },
        background='uniform valid 100 m cells within HUC; known report cells excluded',
        limitations=[
            'not original land-cover/floodplain-matched pseudo-absences',
            'begin locations primary; all location rows sensitivity',
            'historical overlap reconstructed from current NOAA begin records, not released training points',
            'current HUC boundaries simplified to 100 m; reporting and exposure biases remain',
            'descriptive evaluation; 2024–2025 no longer an untouched holdout',
        ],
    )
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
