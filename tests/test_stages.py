import dataclasses
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import icechunk
import numpy as np
import pyarrow.parquet as pq
import pytest
import shapely
import xarray as xr
from geoparquet_lod import validate
from pydantic import ValidationError

from ocr import stages
from ocr.config import OutputPaths, Risk
from ocr.duck import duckdb_connect
from ocr.geoparquet import bbox_sql
from ocr.regions import RegionLevel
from ocr.stages import RISKS, positive, write_buildings, write_regions

# Four blocks in one county, split at (-105.25, 40.0).
BLOCKS = {
    '080130001001000': (-105.4, 39.9, -105.25, 40.0),
    '080130001001001': (-105.25, 39.9, -105.1, 40.0),
    '080130001001002': (-105.4, 40.0, -105.25, 40.1),
    '080130001001003': (-105.25, 40.0, -105.1, 40.1),
}


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(3)
    n = 3000
    x = -105.4 + rng.random(n) * 0.3
    y = 39.9 + rng.random(n) * 0.2
    boxes = shapely.box(x, y, x + 1e-4, y + 1e-4)
    p = {'source': tmp_path / 'source.parquet', 'blocks': tmp_path / 'blocks.parquet'}
    rows = ', '.join(f"('b{i}', '{g.wkt}')" for i, g in enumerate(boxes))
    blocks = ', '.join(f"('{k}', ST_MakeEnvelope{v})" for k, v in BLOCKS.items())
    with duckdb_connect() as con:
        con.sql(f"""
            COPY (SELECT id, geometry, {bbox_sql('geometry')} AS bbox FROM (
                SELECT id, ST_SetCRS(ST_GeomFromText(wkt), 'OGC:CRS84') AS geometry
                FROM (VALUES {rows}) t(id, wkt)))
            TO '{p['source']}' (FORMAT parquet)
        """)
        con.sql(f"""
            COPY (SELECT GEOID, geometry, {bbox_sql('geometry')} AS bbox
                  FROM (VALUES {blocks}) t(GEOID, geometry))
            TO '{p['blocks']}' (FORMAT parquet)
        """)
    return p


def _ds() -> xr.Dataset:
    rng = np.random.default_rng(0)
    attrs = {'crs': 'EPSG:4326'}
    lat, lon = 39.85 + 0.01 * np.arange(30), -105.45 + 0.01 * np.arange(40)
    risk = rng.random((lat.size, lon.size)).astype('float32')
    risk[:, ::3] = 0
    return xr.Dataset(
        {
            'rps_2011': (('latitude', 'longitude'), risk),
            'rps_2047': (('latitude', 'longitude'), risk * 1.5),
        },
        coords={'latitude': ('latitude', lat, attrs), 'longitude': ('longitude', lon, attrs)},
        attrs={'title': 'fire'},
    )


OUT = OutputPaths(Risk.FIRE, 'v2.0.0')


def test_write_buildings_then_regions(
    inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / 'store'
    (root / RegionLevel.BLOCK.key).parent.mkdir(parents=True)
    shutil.copy(inputs['blocks'], root / RegionLevel.BLOCK.key)
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(root))

    write_buildings(
        _ds(),
        OUT,
        blocks=inputs['blocks'],
        source=inputs['source'],
        workers=1,
        log=lambda *_: None,
    )
    buildings = root / 'output/fire/v2.0.0/vector/buildings.geoparquet'
    validate(str(buildings))
    assert [p.name for p in (root / 'output/fire/v2.0.0/vector').iterdir()] == [
        'buildings.geoparquet'
    ]
    tbl = pq.read_table(buildings, columns=['id', 'GEOID', 'rps_2011', 'rps_2047'])
    # A few boxes on the outer block edge have their centroid in no block, so they're dropped.
    assert 2900 < tbl.num_rows <= 3000
    assert tbl['GEOID'].null_count == 0
    assert len(set(tbl['id'].to_pylist())) == tbl.num_rows
    meta = pq.read_schema(buildings).metadata
    assert meta[b'dataset_title'] == b'fire' and b'tiger_vintage' in meta and b'notice' in meta
    assert meta[b'version'] == b'v2.0.0'

    write_regions(OUT, [RegionLevel.BLOCK])
    files = list((root / 'output/fire/v2.0.0/vector/regions').rglob('*.parquet'))
    assert (
        sum(
            pq.read_table(f, columns=['building_count'])['building_count'].to_numpy().sum()
            for f in files
        )
        == tbl.num_rows
    )
    for f in files:
        schema = pq.read_schema(f)
        geo = json.loads(schema.metadata[b'geo'])
        assert 'lod_columns' not in geo  # LOD info stays in the buildings file
        assert b'notice' in schema.metadata and schema.metadata[b'version'] == b'v2.0.0'
        assert {'rps_2011_mean', 'rps_2011_median', 'rps_2011_hist'} <= set(schema.names)


def test_bbox_subsets_buildings_and_regions(
    inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / 'store'
    (root / RegionLevel.BLOCK.key).parent.mkdir(parents=True)
    shutil.copy(inputs['blocks'], root / RegionLevel.BLOCK.key)
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(root))

    west = (-105.4, 39.9, -105.25, 40.1)  # the two western blocks
    write_buildings(
        _ds(),
        OUT,
        blocks=inputs['blocks'],
        source=inputs['source'],
        bbox=west,
        workers=1,
        log=lambda *_: None,
    )
    tbl = pq.read_table(root / 'output/fire/v2.0.0/vector/buildings.geoparquet')
    b = tbl['bbox'].to_pylist()
    assert 0 < tbl.num_rows < 3000
    assert all(west[0] <= r['xmin'] and r['xmax'] <= west[2] for r in b)

    write_regions(OUT, [RegionLevel.BLOCK])
    files = list((root / 'output/fire/v2.0.0/vector/regions').rglob('*.parquet'))
    got = {g for f in files for g in pq.read_table(f, columns=['GEOID'])['GEOID'].to_pylist()}
    assert got == {'080130001001000', '080130001001002'}


def test_positive_excludes_nan_and_null() -> None:
    rows = "(1.0), (0.0), ('NaN'::DOUBLE), (NULL)"
    sql = f'SELECT count(*) FROM (VALUES {rows}) t(x) WHERE {positive("x")}'
    with duckdb_connect() as con:
        assert con.sql(sql).fetchall() == [(1,)]


def test_check_storage_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    out, flood = OutputPaths(Risk.FLOOD, 'v2.0.0'), RISKS[Risk.FLOOD]
    for level in RegionLevel:
        (tmp_path / level.key).parent.mkdir(parents=True)
        (tmp_path / level.key).touch()
    assert stages.check_storage(['etl', 'buildings'], flood, out, False) == []
    assert len(stages.check_storage(['pyramid', 'buildings'], flood, out, False)) == 2
    assert stages.check_storage(['etl', 'buildings', 'regions'], flood, out, False) == []
    assert stages.check_storage(['regions'], flood, out, False)[0].endswith('run buildings first')
    (tmp_path / out.buildings).parent.mkdir(parents=True)
    (tmp_path / out.buildings).touch()
    assert stages.check_storage(['regions'], flood, out, False) == []
    # A risk with score: its etl writes inputs, not the store.
    scored = dataclasses.replace(flood, score=lambda out: None)
    assert stages.check_storage(['etl', 'buildings'], scored, out, False)[0].endswith(
        'run score first'
    )


def test_check_storage_tiger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    out, flood = OutputPaths(Risk.FLOOD, 'v2.0.0'), RISKS[Risk.FLOOD]
    # etl writes the store and buildings the buildings file, so only TIGER is missing.
    problems = stages.check_storage(['etl', 'buildings', 'regions'], flood, out, False)
    assert len(problems) == 1 + len(RegionLevel) and 'regions-etl' in problems[0]
    # Own blocks file, and only the levels asked for.
    kw = {'buildings': {'blocks': 'x.parquet'}, 'regions': {'levels': ['states']}}
    assert stages.check_storage(['etl', 'buildings', 'regions'], flood, out, False, kw) == [
        f'regions needs TIGER states at {RegionLevel.STATE.uri}: run pixi run regions-etl'
    ]


def test_check_storage_etl_existing_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Flood etl refuses an existing store without --overwrite; fail before a job starts.
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    out, flood = OutputPaths(Risk.FLOOD, 'v2.0.0'), RISKS[Risk.FLOOD]
    (tmp_path / out.store).mkdir(parents=True)
    (tmp_path / out.store / 'repo').touch()
    assert stages.check_storage(['etl'], flood, out, False)[0].endswith('skip etl')
    assert stages.check_storage(['etl'], flood, out, True) == []


def test_score_overwrite_deletes_store(monkeypatch: pytest.MonkeyPatch) -> None:
    deleted, scored = [], MagicMock()
    monkeypatch.setattr(stages, '_delete_prefix', deleted.append)
    out = OutputPaths(Risk.FLOOD, 'v2.0.0')
    spec = dataclasses.replace(RISKS[Risk.FLOOD], score=scored)
    stages.score(spec, out, overwrite=False)
    assert deleted == []
    stages.score(spec, out, overwrite=True)
    assert deleted == [out.store]
    scored.assert_called_with(out)


def test_regions_overwrite_deletes_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    deleted, written = [], MagicMock()
    monkeypatch.setattr(stages, '_delete_prefix', deleted.append)
    monkeypatch.setattr(stages, 'write_regions', written)
    out = OutputPaths(Risk.FLOOD, 'v2.0.0')
    stages.regions(RISKS[Risk.FLOOD], out, overwrite=False)
    assert deleted == []
    stages.regions(RISKS[Risk.FLOOD], out, overwrite=True)
    assert deleted == [f'{out.regions}/{level}' for level in RegionLevel]
    written.assert_called_with(out)
    deleted.clear()
    stages.regions(RISKS[Risk.FLOOD], out, overwrite=True, levels=[RegionLevel.COUNTY])
    assert deleted == [f'{out.regions}/counties']  # other levels kept


def test_pyramid_kwargs() -> None:
    kw = stages.stage_kwargs('pyramid', {'levels': 8, 'branch': 'x'})
    assert kw.model_dump(exclude_unset=True) == {'levels': 8, 'branch': 'x'}
    for bad in ({'bogus': 1}, {'overwrite': True}):  # overwrite is --overwrite, not a kwarg
        with pytest.raises(ValidationError):
            stages.stage_kwargs('pyramid', bad)


def test_regions_kwargs() -> None:
    kw = stages.stage_kwargs('regions', {'levels': ['states', 'counties']})
    assert kw.levels == [RegionLevel.STATE, RegionLevel.COUNTY]
    with pytest.raises(ValidationError):
        stages.stage_kwargs('regions', {'levels': ['bogus']})


def test_buildings_bbox_kwarg() -> None:
    kw = stages.stage_kwargs('buildings', {'bbox': [-122.5, 37.7, -122.3, 37.8]})
    assert kw.bbox == (-122.5, 37.7, -122.3, 37.8)
    with pytest.raises(ValidationError):
        stages.stage_kwargs('buildings', {'bbox': [1, 2, 3]})
    for bad in ({'metadata': {}}, {'log': 'print'}):  # set by write_buildings
        with pytest.raises(ValidationError):
            stages.stage_kwargs('buildings', bad)


@pytest.mark.parametrize('risk', list(Risk))
def test_pipeline_from_store(
    risk: Risk, inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every stage after etl, through the CLI on local storage, from a hand-written store.icechunk:
    # checks each stage reads what the previous one wrote (the hand-off Coiled jobs rely on), with
    # real preflight. Each risk's etl, which writes that store, has its own test.
    import xproj  # noqa: F401
    from icechunk.xarray import to_icechunk
    from pyproj import Transformer
    from typer.testing import CliRunner

    from ocr import cli
    from ocr.config import StorageConfig

    root = tmp_path / 'store'
    (root / RegionLevel.BLOCK.key).parent.mkdir(parents=True)
    shutil.copy(inputs['blocks'], root / RegionLevel.BLOCK.key)
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(root))
    spec, out = RISKS[risk], OutputPaths(risk, 'v2.0.0')

    # A 512x512 grid over the blocks in the risk's store CRS, one variable per hist column.
    crs = spec.crs or 'EPSG:4326'
    x0, y1 = Transformer.from_crs(4326, crs, always_xy=True).transform(-105.5, 40.2)
    n, step = 512, (100 if spec.crs else 0.002)
    variables = [v for h in spec.hists for v in h.variables] or ['risk']
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {v: (('y', 'x'), rng.random((n, n), dtype='float32')) for v in variables},
        coords={'y': y1 - step * np.arange(n), 'x': x0 + step * np.arange(n)},
    )
    ds = ds.proj.assign_crs(spatial_ref=crs).proj.write_crs_info()
    repo = icechunk.Repository.create(StorageConfig().icechunk_storage(out.store))
    session = repo.writable_session('main')
    to_icechunk(ds, session, group='0')
    session.commit('test store')

    run = [s for s in ('pyramid', 'buildings', 'regions') if not stages.unimplemented([s], spec)]
    kwargs = {
        'pyramid': {'levels': 2},
        'buildings': {
            'blocks': str(inputs['blocks']),
            'source': str(inputs['source']),
            'workers': 1,
        },
        'regions': {'levels': ['blocks']},
    }
    kwargs = {k: v for k, v in kwargs.items() if k in run}
    args = ['stage', '--version', 'v2.0.0', '--kwargs', json.dumps(kwargs), risk.value]
    r = CliRunner().invoke(cli.app, [*args, *run])
    assert r.exit_code == 0, r.output
    tbl = pq.read_table(root / out.buildings)
    assert 2900 < tbl.num_rows <= 3000
    assert all(tbl[v].null_count == 0 for v in variables if v in tbl.column_names)
    assert list((root / out.regions).rglob('*.parquet'))
