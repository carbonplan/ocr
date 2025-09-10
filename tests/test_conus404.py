import numpy as np
import pyproj
import pytest
import xarray as xr

from ocr import conus404

# ------------------ Helpers ------------------ #


def _make_spatial_constants(time_len: int = 2, y_len: int = 3, x_len: int = 4):
    """Create a dataset for SINALPHA, COSALPHA, crs matching dims of base datasets."""
    shape = (time_len, y_len, x_len)
    sin_alpha = xr.DataArray(
        np.zeros(shape, dtype='float32'), dims=('time', 'y', 'x'), name='SINALPHA'
    )
    cos_alpha = xr.DataArray(
        np.ones(shape, dtype='float32'), dims=('time', 'y', 'x'), name='COSALPHA'
    )

    # Provide a WGS84 CRS WKT (geo_sel relies on crs.attrs['crs_wkt'])
    wkt = pyproj.CRS.from_epsg(4326).to_wkt()
    crs = xr.DataArray(0, name='crs')
    crs.attrs['crs_wkt'] = wkt

    return xr.Dataset({'SINALPHA': sin_alpha, 'COSALPHA': cos_alpha, 'crs': crs})


# ------------------ load_conus404 Tests ------------------ #


def test_load_conus404_with_spatial_constants():
    ds = conus404.load_conus404(add_spatial_constants=True)

    # Variables present
    for v in ['PSFC', 'Q2', 'T2', 'TD2', 'U10', 'V10', 'SINALPHA', 'COSALPHA']:
        assert v in ds

    # lat/lon re-added
    assert 'lat' in ds.coords and 'lon' in ds.coords

    # crs is a coordinate and has wkt
    assert 'crs' in ds.coords
    assert 'crs_wkt' in ds.crs.attrs


def test_load_conus404_without_spatial_constants():
    ds = conus404.load_conus404(add_spatial_constants=False)

    # core vars only
    for v in ['PSFC', 'Q2', 'T2', 'TD2', 'U10', 'V10']:
        assert v in ds
    for v in ['SINALPHA', 'COSALPHA']:
        assert v not in ds

    # lat/lon present
    assert 'lat' in ds.coords and 'lon' in ds.coords


# ------------------ Meteorological Calculations ------------------ #


def test_compute_relative_humidity():
    # Provide temps and dewpoint close together to yield high RH
    T2 = xr.DataArray(
        273.15 + np.array([10.0, 12.0, 14.0]), dims=('time',), name='T2', attrs={'units': 'K'}
    )
    TD2 = xr.DataArray(
        273.15 + np.array([9.0, 11.0, 13.0]), dims=('time',), name='TD2', attrs={'units': 'K'}
    )
    Q2 = xr.DataArray(
        np.linspace(0.002, 0.004, 3), dims=('time',), name='Q2', attrs={'units': 'kg/kg'}
    )  # unused currently
    PSFC = xr.DataArray(
        np.full(3, 101325), dims=('time',), name='PSFC', attrs={'units': 'Pa'}
    )  # unused currently

    ds = xr.Dataset({'T2': T2, 'TD2': TD2, 'Q2': Q2, 'PSFC': PSFC})
    hurs = conus404.compute_relative_humidity(ds)
    assert isinstance(hurs, xr.DataArray)
    assert hurs.name == 'hurs'
    assert (hurs >= 0).all() and (hurs <= 100).all()


def test_rotate_winds_to_earth():
    U10 = xr.DataArray(np.ones(2), dims=('time',), name='U10')
    V10 = xr.DataArray(np.ones(2) * 2, dims=('time',), name='V10')
    # simple 45 deg rotation: sin=0, cos=1 => earth winds unchanged
    SINALPHA = xr.DataArray(np.zeros(2), dims=('time',), name='SINALPHA')
    COSALPHA = xr.DataArray(np.ones(2), dims=('time',), name='COSALPHA')
    ds = xr.Dataset({'U10': U10, 'V10': V10, 'SINALPHA': SINALPHA, 'COSALPHA': COSALPHA})

    u_earth, v_earth = conus404.rotate_winds_to_earth(ds)
    assert np.allclose(u_earth, U10)
    assert np.allclose(v_earth, V10)
    assert u_earth.name == 'u10_earth' and v_earth.name == 'v10_earth'


def test_compute_wind_speed_and_direction():
    u = xr.DataArray([3.0, 0.0, -4.0], dims=('time',), name='u10_earth', attrs={'units': 'm/s'})
    v = xr.DataArray([4.0, -5.0, 0.0], dims=('time',), name='v10_earth', attrs={'units': 'm/s'})
    ds = conus404.compute_wind_speed_and_direction(u, v)

    # Expect speed sqrt(u^2+v^2)
    expected_speed = np.sqrt(u.values**2 + v.values**2)
    assert 'sfcWind' in ds and 'sfcWindfromdir' in ds
    assert np.allclose(ds['sfcWind'], expected_speed)


# ------------------ geo_sel Tests ------------------ #


def _make_geo_dataset():
    # Create a small projected grid (use EPSG:4326 to keep transforms trivial)
    y = xr.DataArray([35.0, 36.0], dims=('y',), name='y')
    x = xr.DataArray([-120.0, -119.0], dims=('x',), name='x')
    data = xr.DataArray(np.random.rand(2, 2), dims=('y', 'x'), name='dummy')
    wkt = pyproj.CRS.from_epsg(4326).to_wkt()
    crs = xr.DataArray(0, name='crs')
    crs.attrs['crs_wkt'] = wkt
    ds = xr.Dataset({'dummy': data, 'crs': crs}, coords={'y': y, 'x': x})
    ds = ds.set_coords(['crs'])  # mimic load_conus404 behavior
    return ds


def test_geo_sel_point():
    ds = _make_geo_dataset()
    # Choose point near first grid cell
    out = conus404.geo_sel(ds, lon=-119.2, lat=35.2, method='nearest')
    assert 'dummy' in out
    assert out.dims.get('x') is None and out.dims.get('y') is None  # reduced to scalars
    assert out.attrs['requested_lon'] == -119.2
    assert out.attrs['requested_lat'] == 35.2


def test_geo_sel_bbox():
    ds = _make_geo_dataset()
    # Bbox that spans entire domain
    out = conus404.geo_sel(ds, bbox=(-121.0, 34.5, -118.5, 36.5))
    assert out.dims['x'] == 2 and out.dims['y'] == 2


def test_geo_sel_invalid_conflict():
    ds = _make_geo_dataset()
    with pytest.raises(ValueError, match='Provide either bbox OR point'):
        conus404.geo_sel(ds, lon=-119, lat=35, bbox=(-121, 34, -118, 36))


def test_geo_sel_invalid_missing():
    ds = _make_geo_dataset()
    with pytest.raises(ValueError, match='You must supply'):
        conus404.geo_sel(ds)
