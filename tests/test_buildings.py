from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pytest
import xarray as xr

from ocr.buildings import _cell, _grid, _sample, sample_buildings, write_patable_to_gpq


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
    iy, ix = np.array([0, 3, -1, 2]), np.array([0, 4, 0, 5])
    got = _sample(ds, iy, ix)
    assert got['risk'].to_pylist() == [0, 19, None, None]
    assert got['cls'].to_pylist() == [0, 19, None, None]
    assert got['cls'].type == pa.int8()  # off-raster is null, not a cast to float


def test_grid() -> None:
    assert _grid(np.array([22.0, 22.5, 23.0])) == (22.0, 0.5)
    step = np.float32(0.00030799387843481654)
    _grid(np.float32(22.43) + step * np.arange(100_000, dtype='float32'))  # 30m float32
    with pytest.raises(ValueError, match='evenly spaced'):
        _grid(np.array([0.0, 1.0, 3.0]))


def test_rejects_bad_inputs_before_s3() -> None:
    with pytest.raises(ValueError, match='not EPSG:4326'):
        sample_buildings(_ds('EPSG:5070'))
    ds = _ds()
    ds['risk3d'] = ds['risk'].expand_dims(time=2)
    with pytest.raises(ValueError, match='not 2D'):
        sample_buildings(ds)
    with pytest.raises(ValueError, match='not a valid PointSamplingStrategy'):
        sample_buildings(_ds(), point_strategy='1; DROP TABLE x')


def test_write_geoparquet_metadata(tmp_path: Path) -> None:
    meta = {'overture_release': '2026-09-23.1', "dataset_it's": "it's"}
    con = duckdb.connect()
    con.sql('INSTALL spatial; LOAD spatial;')
    tbl = con.sql(
        "SELECT * FROM (VALUES ('a', ST_Point(0, 0)), ('b', ST_Point(1, 1))) t(id, geometry)"
    ).to_arrow_table()
    tbl = tbl.append_column('cls', pa.array([3, None], type=pa.int8()))
    out = tmp_path / 'out.parquet'
    write_patable_to_gpq(tbl.replace_schema_metadata(meta), out)
    # Off-raster null in an int var survives the write as a null int, not a float NaN.
    assert con.sql(f"SELECT cls FROM '{out}' ORDER BY id").fetchall() == [(3,), (None,)]
    assert con.sql(f"SELECT typeof(cls) FROM '{out}' LIMIT 1").fetchall() == [('TINYINT',)]
    rows = dict(
        con.sql(f"SELECT decode(key), decode(value) FROM parquet_kv_metadata('{out}')").fetchall()
    )
    assert meta.items() <= rows.items()
    assert 'geo' in rows


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
        f'SELECT {_cell("y", _grid(lat))} AS iy, {_cell("x", _grid(lon))} AS ix FROM pts'
    ).fetchnumpy()
    iy, ix = idx['iy'], idx['ix']
    got = _sample(ds, iy, ix)['risk'].to_numpy()
    expected = ds['risk'].sel(
        latitude=xr.DataArray(y, dims='pt'), longitude=xr.DataArray(x, dims='pt'), method='nearest'
    )
    np.testing.assert_array_equal(got, expected.values)
