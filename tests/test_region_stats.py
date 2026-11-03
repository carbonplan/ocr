import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ocr.config import OutputPaths, Risk
from ocr.duck import duckdb_connect
from ocr.geoparquet import bbox_sql
from ocr.regions import TIGER_VINTAGE, RegionLevel
from ocr.regions.layout import write_level
from ocr.regions.stats import Hist, write_region_stats
from ocr.regions.tiger import nation_sql
from ocr.stages import RISKS, RiskOutputs, write_regions

BINS = [0.01, 0.1, 1, 100]
HISTS = [Hist(['v'], BINS)]
A, B, C = (
    '060010001001000',
    '060010001001001',
    '060030001001000A',
)  # A, B in 06001; C (suffixed block) in 06003


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    p = {k: tmp_path / f'{k}.parquet' for k in ['buildings', 'blocks', 'counties', 'nation']}
    with duckdb_connect() as con:
        # Unit squares: A at (0,0), B at (1,0), C at (0,1). Counties: 06001 = A+B, 06003 = C.
        # Blocks have no NAME, as written by the ETL.
        con.sql(f"""
            COPY (SELECT *, {bbox_sql('geometry')} AS bbox FROM (VALUES
                ('{A}', ST_MakeEnvelope(0, 0, 1, 1)),
                ('{B}', ST_MakeEnvelope(1, 0, 2, 1)),
                ('{C}', ST_MakeEnvelope(0, 1, 1, 2))) t(GEOID, geometry))
            TO '{p['blocks']}' (FORMAT parquet)
        """)
        con.sql(f"""
            COPY (SELECT *, {bbox_sql('geometry')} AS bbox FROM (VALUES
                ('06001', 'Alameda', ST_MakeEnvelope(0, 0, 2, 1)),
                ('06003', 'Alpine', ST_MakeEnvelope(0, 1, 1, 2)),
                ('06005', 'Amador', ST_MakeEnvelope(9, 9, 10, 10))) t(GEOID, NAME, geometry))
            TO '{p['counties']}' (FORMAT parquet)
        """)
        con.sql(f"COPY ({nation_sql(str(p['counties']))}) TO '{p['nation']}' (FORMAT parquet)")
        # Tagged buildings (as sample_buildings writes them): tiny squares centred on (x, y) in block GEOID.
        con.sql(f"""
            COPY (
                SELECT id, GEOID, v, ST_MakeEnvelope(x - 0.0078125, y - 0.0078125, x + 0.0078125, y + 0.0078125) AS geometry,
                       {{'xmin': x, 'ymin': y, 'xmax': x, 'ymax': y}} AS bbox
                FROM (VALUES
                    ('zero', '{A}', 0.5, 0.5, 0.0),
                    ('low', '{A}', 0.25, 0.25, 0.015),
                    ('b', '{B}', 1.5, 0.5, 0.05),
                    ('neg', '{B}', 1.5, 0.25, -1.0),          -- below 0
                    ('mid', '{A}', 0.75, 0.5, 0.5),
                    ('over', '{C}', 0.5, 1.5, 150.0),         -- above the last bin
                    ('nan', '{C}', 0.5, 1.6, 'NaN'::DOUBLE),
                    ('outside', NULL, 5.0, 5.0, 1.0)) t(id, GEOID, x, y, v))
            TO '{p['buildings']}' (FORMAT parquet, KV_METADATA {{overture_release: 'r1'}})
        """)
    return p


def _stats(paths: dict[str, Path], level: RegionLevel, regions: Path, hists=HISTS) -> pa.Table:
    out = paths['buildings'].with_name('stats.parquet')
    write_region_stats(paths['buildings'], level, hists, out, regions)
    return pq.read_table(out)


def _rows(paths: dict[str, Path], level: RegionLevel, regions: str) -> dict[str, dict]:
    tbl = _stats(paths, level, paths[regions])
    return {r['GEOID']: r for r in tbl.to_pylist()}


def _rows_with(paths: dict[str, Path], hists: list[Hist]) -> dict[str, dict]:
    tbl = _stats(paths, RegionLevel.BLOCK, paths['blocks'], hists)
    return {r['GEOID']: r for r in tbl.to_pylist()}


def test_block_stats(paths) -> None:
    rows = _rows(paths, RegionLevel.BLOCK, 'blocks')
    assert list(rows) == [A, B, C]
    a = rows[A]
    assert a['building_count'] == 3
    assert a['v_mean'] == pytest.approx((0 + 0.015 + 0.5) / 3)
    assert a['v_median'] == pytest.approx(0.015)
    assert a['v_hist'] == [1, 0, 1, 1, 0]  # [zeros, <=0.01, <=0.1, <=1, <=100]
    assert a['bbox'] == {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1}


def test_negative_in_first_bin(paths) -> None:
    b = _rows(paths, RegionLevel.BLOCK, 'blocks')[B]
    assert b['building_count'] == 2
    assert b['v_mean'] == pytest.approx((0.05 - 1) / 2)
    assert b['v_hist'] == [0, 1, 1, 0, 0]  # -1 in the first nonzero bin, as in fire


def test_overflow_in_last_bin_and_nan_missing(paths) -> None:
    c = _rows(paths, RegionLevel.BLOCK, 'blocks')[C]
    assert c['building_count'] == 2  # NaN building still counts
    assert c['v_mean'] == 150.0  # NaN ignored
    assert c['v_hist'] == [0, 0, 0, 0, 1]  # 150 > last edge, in the last bin


def test_nan_missing_when_pyarrow_written(paths) -> None:
    # pyarrow's float stats omit NaN; DuckDB must not take them as proof of no NaN (duckdb#25521).
    pq.write_table(pq.read_table(paths['buildings']), paths['buildings'])
    c = _rows(paths, RegionLevel.BLOCK, 'blocks')[C]
    assert c['v_mean'] == 150.0
    assert c['v_median'] == 150.0


def test_hist_plus_missing_is_building_count(paths) -> None:
    for level, regions in [(RegionLevel.BLOCK, 'blocks'), (RegionLevel.COUNTY, 'counties')]:
        for r in _rows(paths, level, regions).values():
            assert r['v_missing'] == (r['GEOID'] in (C, '06003'))  # the 'nan' building
            assert sum(r['v_hist']) + r['v_missing'] == r['building_count']


def test_no_hists(paths) -> None:
    # Every value column gets mean and median; only Hist vars get a hist.
    rows = _rows_with(paths, [])
    assert rows[A]['building_count'] == 3
    assert [c for c in rows[A] if c.startswith('v_')] == ['v_mean', 'v_median', 'v_missing']
    assert rows[A]['v_median'] == pytest.approx(0.015)


def test_hist_variable_missing(paths) -> None:
    with pytest.raises(ValueError, match='not in buildings'):
        _stats(paths, RegionLevel.BLOCK, paths['blocks'], [Hist(['w'], BINS)])


def test_variable_in_two_hists(paths) -> None:
    with pytest.raises(ValueError, match='more than one Hist'):
        _stats(paths, RegionLevel.BLOCK, paths['blocks'], [Hist(['v'], BINS), Hist(['v'], [1])])


def test_county_rollup_by_prefix(paths) -> None:
    rows = _rows(paths, RegionLevel.COUNTY, 'counties')
    assert list(rows) == ['06001', '06003']  # Amador has no buildings
    assert rows['06001']['NAME'] == 'Alameda'
    assert rows['06001']['building_count'] == 5
    assert rows['06001']['v_hist'] == [1, 1, 2, 1, 0]


def test_nation(paths) -> None:
    (row,) = _stats(paths, RegionLevel.NATION, paths['nation']).to_pylist()
    assert row['GEOID'] == 'US'
    assert row['NAME'] == 'United States'
    assert row['building_count'] == 7  # every tagged building; 'outside' has no block
    assert row['v_hist'] == [1, 1, 2, 1, 1]


@pytest.mark.parametrize('bins', [[], [0, 1], [1, 1], [2, 1], [1, float('inf')], [float('nan')]])
def test_bad_bins(bins: list[float]) -> None:
    with pytest.raises(ValueError):
        Hist(['v'], bins)


def test_write_regions(paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / 'store'
    levels = [RegionLevel.BLOCK, RegionLevel.COUNTY, RegionLevel.NATION]
    for level, src in zip(
        levels, [paths['blocks'], paths['counties'], paths['nation']], strict=True
    ):
        dst = root / level.key
        dst.parent.mkdir(parents=True)
        shutil.copy(src, dst)
    (root / 'output/flood/v2.0.0/vector').mkdir(parents=True)
    shutil.copy(paths['buildings'], root / 'output/flood/v2.0.0/vector/buildings.geoparquet')
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(root))
    monkeypatch.setitem(RISKS, Risk.FLOOD, RiskOutputs(HISTS, 'true', []))
    # temp_dir is used: a missing one fails, an existing one is left empty.
    with pytest.raises(FileNotFoundError):
        write_regions(OutputPaths(Risk.FLOOD, 'v2.0.0'), levels, temp_dir=tmp_path / 'missing')
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    write_regions(OutputPaths(Risk.FLOOD, 'v2.0.0'), levels, temp_dir=scratch)
    assert not any(scratch.iterdir())
    out = root / 'output/flood/v2.0.0/vector/regions'
    stats = ['building_count', 'v_median', 'v_hist', 'v_mean', 'v_missing']
    files = {
        'blocks/state_fips=06/county_fips=001/data.parquet': (2, []),
        'blocks/state_fips=06/county_fips=003/data.parquet': (1, []),
        'counties/state_fips=06/data.parquet': (2, ['NAME']),
        'nation.parquet': (1, ['NAME']),
    }
    assert sorted(str(p.relative_to(out)) for p in out.rglob('*.parquet')) == sorted(files)
    for rel, (n, extra) in files.items():
        f = pq.ParquetFile(out / rel)
        meta = f.schema_arrow.metadata
        assert f.metadata.num_rows == n
        assert json.loads(meta[b'bins']) == {'v': BINS}
        assert meta[b'tiger_vintage'] == TIGER_VINTAGE.encode()
        assert meta[b'overture_release'] == b'r1'  # buildings' metadata carried through
        geo = json.loads(meta[b'geo'])
        assert geo['columns']['geometry']['covering']['bbox']['xmin'] == ['bbox', 'xmin']
        assert f.schema_arrow.names == ['GEOID', *extra, *stats, 'bbox', 'geometry']
        assert f.metadata.row_group(0).column(0).statistics.has_min_max
        assert f.metadata.row_group(0).column(0).has_offset_index


def test_hist_coerces_numpy_bins(paths) -> None:
    h = Hist(['v'], np.array(BINS))
    assert h.bins == tuple(BINS) and type(h.bins[0]) is float
    got = _stats(paths, RegionLevel.BLOCK, paths['blocks'], [h]).to_pylist()
    assert got[0]['v_hist'] == _rows(paths, RegionLevel.BLOCK, 'blocks')[A]['v_hist']


def test_nation_sql_dissolves(tmp_path: Path) -> None:
    states = tmp_path / 'states.parquet'
    with duckdb_connect() as con:
        con.sql(f"""
            COPY (SELECT * FROM (VALUES
                ('01', ST_MakeEnvelope(0, 0, 1, 1)),
                ('04', ST_MakeEnvelope(1, 0, 2, 1))) t(GEOID, geometry))
            TO '{states}' (FORMAT parquet)
        """)
        rows = con.sql(
            f'SELECT GEOID, ST_Area(geometry) FROM ({nation_sql(str(states))})'
        ).fetchall()
    assert rows == [('US', 2.0)]


def _page_first_rows(path: Path, column: str) -> list[int]:
    """First row of each page of `column` in row group 0, from the offset index."""
    # pyarrow doesn't expose the offset index; geoparquet-lod's Thrift reader does (pinned git rev).
    # TODO: _compact is private to geoparquet-lod; ask upstream to expose an offset index reader and switch to it.
    from geoparquet_lod._compact import get, loads

    data = path.read_bytes()
    footer = int.from_bytes(data[-8:-4], 'little')
    group = get(loads(data[-8 - footer : -8]), 4)[1][0]
    for chunk in get(group, 1)[1]:
        if '.'.join(p.decode() for p in get(get(chunk, 3), 3)[1]) == column:
            offset, length = get(chunk, 4), get(chunk, 5)
            return [get(page, 3) for page in get(loads(data[offset : offset + length]), 1)[1]]
    raise KeyError(column)


def test_write_level_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 70 blocks in one county: 32 rows per page, so pages start at rows 0, 32, 64 in every column.
    src = tmp_path / 'blocks.parquet'
    with duckdb_connect() as con:
        con.sql(f"""
            COPY (SELECT GEOID, building_count, {bbox_sql('geometry')} AS bbox, geometry FROM (
                SELECT '0603710111' || lpad(i::VARCHAR, 4, '0') AS GEOID, i AS building_count,
                       ST_MakeEnvelope(i, 0, i + 1, 1) AS geometry
                FROM range(70) t(i) ORDER BY random()))
            TO '{src}' (FORMAT parquet)
        """)
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path / 'store'))
    write_level(src, RegionLevel.BLOCK, 'regions')
    out = tmp_path / 'store/regions/blocks/state_fips=06/county_fips=037/data.parquet'
    geoids = pq.read_table(out, columns=['GEOID'])['GEOID'].to_pylist()
    assert len(geoids) == 70 and geoids == sorted(geoids)
    for column in ['GEOID', 'building_count', 'bbox.xmin', 'geometry']:
        assert _page_first_rows(out, column) == [0, 32, 64]
