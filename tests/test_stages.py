import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import shapely
import xarray as xr
from geoparquet_lod import validate

from ocr.config import OutputPaths, Risk
from ocr.duck import duckdb_connect
from ocr.geoparquet import bbox_sql
from ocr.regions import RegionLevel
from ocr.stages import positive, write_buildings, write_regions

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


def test_positive_excludes_nan_and_null() -> None:
    rows = "(1.0), (0.0), ('NaN'::DOUBLE), (NULL)"
    sql = f'SELECT count(*) FROM (VALUES {rows}) t(x) WHERE {positive("x")}'
    with duckdb_connect() as con:
        assert con.sql(sql).fetchall() == [(1,)]
