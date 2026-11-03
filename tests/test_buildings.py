from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
import xarray as xr
import xproj  # noqa: F401

from ocr.buildings import (
    OVERTURE_RELEASE,
    VECTOR_METADATA,
    PointSamplingStrategy,
    _axis,
    _cell_index_sql,
    _metadata,
    _sample,
    _validate_dataset,
    sample_buildings,
)
from ocr.duck import duckdb_connect
from ocr.geoparquet import bbox_sql, write_geoparquet
from ocr.regions import TIGER_VINTAGE


def _ds(crs: str = 'EPSG:4326') -> xr.Dataset:
    # Ascending latitude with crs attrs on both coords, like the fire store.
    attrs = {'crs': crs}
    return xr.Dataset(
        {'risk': (('latitude', 'longitude'), np.arange(20, dtype='float32').reshape(4, 5))},
        coords={
            'latitude': ('latitude', np.arange(4) + 30.5, attrs),
            'longitude': ('longitude', np.arange(5) - 99.5, attrs),
        },
    )


def test_sample_inside_and_out_of_bounds() -> None:
    ds = _ds()
    ds['cls'] = ds['risk'].astype('int8')
    ds['risk'][0, 1] = np.nan
    iy, ix = np.array([0, 3, -1, 2, 0]), np.array([0, 4, 0, 5, 1])
    got = _sample(ds, iy, ix)
    assert got['risk'].to_pylist() == [0, 19, None, None, None]  # NaN pixel is null too
    assert got['cls'].to_pylist() == [0, 19, None, None, 1]
    assert got['cls'].type == pa.int8()  # off-raster is null, not a cast to float


def test_axis() -> None:
    assert _axis(np.array([22.0, 22.5, 23.0])) == (22.0, 0.5)
    step = np.float32(0.00030799387843481654)
    _axis(np.float32(22.43) + step * np.arange(100_000, dtype='float32'))  # 30m float32
    with pytest.raises(ValueError, match='evenly spaced'):
        _axis(np.array([0.0, 1.0, 3.0]))


def test_validate_dataset_returns_axes() -> None:
    assert _validate_dataset(_ds()) == ('EPSG:4326', (30.5, 1.0), (-99.5, 1.0))


def test_rejects_bad_inputs_before_s3(tmp_path: Path) -> None:
    out = tmp_path / 'out.parquet'
    with pytest.raises(ValueError, match='no CRS'):
        sample_buildings(_ds().drop_attrs(), out)
    ds = _ds()
    ds['risk3d'] = ds['risk'].expand_dims(time=2)
    with pytest.raises(ValueError, match='not 2D'):
        sample_buildings(ds, out)
    with pytest.raises(ValueError, match='not a valid PointSamplingStrategy'):
        sample_buildings(_ds(), out, point_strategy='1; DROP TABLE x')


def test_write_geoparquet_metadata(tmp_path: Path) -> None:
    meta = {'overture_release': '2026-09-23.1', "dataset_it's": "it's"}
    con = duckdb_connect()
    tbl = con.sql(
        "SELECT * FROM (VALUES ('a', ST_Point(0, 0)), ('b', ST_Point(1, 1))) t(id, geometry)"
    ).to_arrow_table()
    tbl = tbl.append_column('cls', pa.array([3, None], type=pa.int8()))
    out = tmp_path / 'vector' / 'out.parquet'  # missing parent dir is created
    write_geoparquet(tbl.replace_schema_metadata(meta), out)
    # Off-raster null in an int var survives the write as a null int, not a float NaN.
    assert con.sql(f"SELECT cls FROM '{out}' ORDER BY id").fetchall() == [(3,), (None,)]
    assert con.sql(f"SELECT typeof(cls) FROM '{out}' LIMIT 1").fetchall() == [('TINYINT',)]
    rows = dict(
        con.sql(f"SELECT decode(key), decode(value) FROM parquet_kv_metadata('{out}')").fetchall()
    )
    assert meta.items() <= rows.items()
    assert 'geo' in rows
    assert not list(tmp_path.glob('*.tmp'))  # renamed into place
    write_geoparquet(tbl, out)  # no metadata at all
    assert con.sql(f"SELECT count(*) FROM '{out}'").fetchall() == [(2,)]
    con.close()


def test_sample_matches_sel_nearest() -> None:
    # Index math runs the same SQL as sample_buildings. float64 coords, like the fire store;
    # float32 coords drift off the even grid, so .sel would differ by one cell near edges.
    step = 0.00030799387843481654  # 30m
    lat = 30.0 + step * np.arange(400)
    lon = -100.0 + step * np.arange(500)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {'risk': (('latitude', 'longitude'), rng.random((lat.size, lon.size)))},
        coords={'latitude': lat, 'longitude': lon},
    )
    y = rng.uniform(lat[0], lat[-1], 10_000)
    x = rng.uniform(lon[0], lon[-1], 10_000)
    con = duckdb.connect()
    con.register('pts', pa.table({'y': y, 'x': x}))
    idx = con.sql(
        f'SELECT {_cell_index_sql("y", _axis(lat))} AS iy, {_cell_index_sql("x", _axis(lon))} AS ix FROM pts'
    ).fetchnumpy()
    con.close()
    iy, ix = idx['iy'], idx['ix']
    got = _sample(ds, iy, ix)['risk'].to_numpy()
    expected = ds['risk'].sel(
        latitude=xr.DataArray(y, dims='pt'), longitude=xr.DataArray(x, dims='pt'), method='nearest'
    )
    np.testing.assert_array_equal(got, expected.values)


def test_metadata() -> None:
    ds = _ds()
    ds.attrs = {'title': 'fire', 'version': 2}
    meta = _metadata(ds, OVERTURE_RELEASE, PointSamplingStrategy.ON_SURFACE)
    assert VECTOR_METADATA.items() <= meta.items()
    assert 'Open Database License' in meta['notice']
    assert meta['overture_release'] == OVERTURE_RELEASE
    assert meta['overture_license'] == 'ODbL-1.0'
    assert meta['dataset_title'] == 'fire'
    assert meta['dataset_version'] == '2'
    assert meta['point_strategy'] == 'ST_PointOnSurface'
    assert meta['tiger_vintage'] == TIGER_VINTAGE


@pytest.fixture
def blocks(tmp_path: Path) -> Path:
    # TIGER is EPSG:4269, Overture CRS84; sample_buildings relabels blocks so DuckDB intersects them.
    out = tmp_path / 'blocks.parquet'
    with duckdb_connect() as con:
        con.sql(f"""
            COPY (SELECT GEOID, ST_SetCRS(geometry, 'EPSG:4269') AS geometry FROM (VALUES
                ('A', ST_MakeEnvelope(-100, 30, -99, 31)),
                ('B', ST_MakeEnvelope(-99, 30, -98, 31)),
                ('C', ST_MakeEnvelope(-100, 31, -99, 32))) t(GEOID, geometry))
            TO '{out}' (FORMAT parquet)
        """)
    return out


def _source(tmp_path: Path, geoms: dict[str, shapely.Geometry]) -> Path:
    """Overture-like buildings: id, CRS84 geometry, bbox."""
    out = tmp_path / 'source.parquet'
    rows = ', '.join(f"('{k}', '{g.wkt}')" for k, g in geoms.items())
    with duckdb_connect() as con:
        con.sql(f"""
            COPY (SELECT id, geometry, {bbox_sql('geometry')} AS bbox FROM (
                SELECT id, ST_SetCRS(ST_GeomFromText(wkt), 'OGC:CRS84') AS geometry
                FROM (VALUES {rows}) t(id, wkt)))
            TO '{out}' (FORMAT parquet)
        """)
    return out


def _square(x: float, y: float) -> shapely.Geometry:
    return shapely.box(x - H, y - H, x + H, y + H)


H = 0.0078125  # 1/128 half-width keeps centroids exact


def _sampled(tmp_path: Path, **kwargs) -> pa.Table:
    out = tmp_path / 'sampled.parquet'
    sample_buildings(_ds(), out, temp_dir=tmp_path, **kwargs)
    assert not list(tmp_path.glob('tmp*'))  # staging files cleaned up
    return pq.read_table(out)


def test_sample_buildings_tags_blocks(tmp_path: Path, blocks: Path) -> None:
    src = _source(
        tmp_path,
        {
            'a': _square(-99.5, 30.5),
            'b': _square(-98.5, 30.5),
            'c': _square(-99.5, 31.5),
            'out': _square(-96.5, 32.5),  # on the raster, in no block: dropped
        },
    )
    tbl = _sampled(tmp_path, blocks=blocks, source=src)
    assert tbl.column_names == ['id', 'GEOID', 'risk', 'geometry', 'bbox']
    rows = {r['id']: (r['GEOID'], r['risk']) for r in tbl.drop_columns(['geometry']).to_pylist()}
    assert rows == {'a': ('A', 0), 'b': ('B', 1), 'c': ('C', 5)}
    assert tbl.schema.metadata[b'tiger_vintage'] == TIGER_VINTAGE.encode()
    assert b'notice' in tbl.schema.metadata


def test_sample_buildings_values_follow_rows(tmp_path: Path) -> None:
    # One building per raster cell, split over two blocks, so the join returns rows in no set order;
    # the positional join must still put each cell's value on its own building.
    blocks = tmp_path / 'blocks.parquet'
    with duckdb_connect() as con:
        con.sql(f"""
            COPY (SELECT * FROM (VALUES
                ('L', ST_MakeEnvelope(-100, 30, -97, 34)),
                ('R', ST_MakeEnvelope(-97, 30, -95, 34))) t(GEOID, geometry))
            TO '{blocks}' (FORMAT parquet)
        """)
    cells = {f'{i}{j}': _square(-99.5 + j, 30.5 + i) for i in range(4) for j in range(5)}
    tbl = _sampled(tmp_path, blocks=blocks, source=_source(tmp_path, cells))
    got = {
        r['id']: (r['GEOID'], r['risk']) for r in tbl.select(['id', 'GEOID', 'risk']).to_pylist()
    }
    assert got == {
        f'{i}{j}': ('L' if j < 3 else 'R', i * 5 + j) for i in range(4) for j in range(5)
    }


def test_sample_buildings_point_strategy(tmp_path: Path) -> None:
    # L-shape: centroid (~(-99.17, 30.83)) falls in the notch, block B; on-surface point is in A.
    blocks = tmp_path / 'blocks.parquet'
    with duckdb_connect() as con:
        con.sql(f"""
            COPY (SELECT * FROM (VALUES
                ('A', ST_GeomFromText('POLYGON((-100 30, -98 30, -98 30.5, -99.5 30.5, -99.5 32, -100 32, -100 30))')),
                ('B', ST_MakeEnvelope(-99.5, 30.5, -98, 32))) t(GEOID, geometry))
            TO '{blocks}' (FORMAT parquet)
        """)
    ell = shapely.Polygon(
        [(-99.9, 30.1), (-98.1, 30.1), (-98.1, 30.4), (-99.6, 30.4), (-99.6, 31.9), (-99.9, 31.9)]
    )
    assert shapely.within(ell.centroid, shapely.box(-99.5, 30.5, -98, 32))
    src = _source(tmp_path, {'l': ell})
    got = _sampled(
        tmp_path, blocks=blocks, source=src, point_strategy=PointSamplingStrategy.ON_SURFACE
    )
    assert got['GEOID'].to_pylist() == ['A']


def test_sample_buildings_projected_raster(tmp_path: Path, blocks: Path) -> None:
    # EPSG:5070 grid (CF crs_wkt like the flood store, descending y), 1 km cells centred on building a's projected centroid.
    with duckdb_connect() as con:
        ((x0, y0),) = con.sql(
            "SELECT ST_X(p), ST_Y(p) FROM (SELECT ST_Transform(ST_Point(-99.5, 30.5), 'OGC:CRS84', 'EPSG:5070', true) p)"
        ).fetchall()
    ds = (
        xr.Dataset(
            {'fdp': (('y', 'x'), np.arange(9, dtype='float32').reshape(3, 3))},
            coords={'y': y0 + np.array([1000.0, 0, -1000]), 'x': x0 + np.array([-1000.0, 0, 1000])},
        )
        .proj.assign_crs(spatial_ref='EPSG:5070')
        .proj.write_crs_info()
    )
    src = _source(
        tmp_path, {'a': _square(-99.5, 30.5), 'b': _square(-98.5, 30.5)}
    )  # b is ~96 km east: off
    out = tmp_path / 'sampled.parquet'
    sample_buildings(ds, out, blocks=blocks, source=src, temp_dir=tmp_path)
    got = pq.read_table(out, columns=['id', 'fdp']).to_pylist()
    assert sorted(got, key=lambda r: r['id']) == [{'id': 'a', 'fdp': 4.0}, {'id': 'b', 'fdp': None}]


def test_sample_buildings_shared_edge_lowest_geoid(tmp_path: Path, blocks: Path) -> None:
    # A centroid exactly on the A|B edge matches both blocks; the lowest GEOID wins, one row per id.
    src = _source(tmp_path, {'edge': _square(-99.0, 30.5), 'a': _square(-99.5, 30.5)})
    tbl = _sampled(tmp_path, blocks=blocks, source=src)
    got = dict(zip(tbl['id'].to_pylist(), tbl['GEOID'].to_pylist(), strict=True))
    assert got == {'edge': 'A', 'a': 'A'}
