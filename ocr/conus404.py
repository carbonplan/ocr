from typing import cast

import xarray as xr
from xclim import convert


def load_conus404(add_spatial_constants: bool = True) -> xr.Dataset:
    """
    Load the CONUS 404 dataset.

    Parameters
    ----------
    add_spatial_constants : bool, optional
        If True, adds spatial constant variables (SINALPHA, COSALPHA) to the dataset.

    Returns
    -------
    ds : xr.Dataset
        The CONUS 404 dataset.
    """

    from ocr import catalog

    variables = ['PSFC', 'Q2', 'T2', 'TD2', 'U10', 'V10']
    dsets = []
    for variable in variables:
        dset = (
            catalog.get_dataset(f'conus404-hourly-{variable}').to_xarray().drop_vars(['lat', 'lon'])
        )
        dsets.append(dset)
    ds = xr.merge(dsets)

    if add_spatial_constants:
        INPUT_ZARR_STORE_CONFIG = {
            'url': 's3://hytest/conus404/conus404_hourly.zarr',
            'storage_options': {
                'anon': True,
                'client_kwargs': {'endpoint_url': 'https://usgs.osn.mghpcc.org/'},
            },
        }

        variables = ['SINALPHA', 'COSALPHA', 'crs']
        spatial_constant_ds = xr.open_dataset(
            INPUT_ZARR_STORE_CONFIG['url'],
            storage_options=INPUT_ZARR_STORE_CONFIG['storage_options'],
            engine='zarr',
            chunks={},
        )[variables]

        ds = xr.merge([ds, spatial_constant_ds])

    # add lat, lon back in
    dset = catalog.get_dataset('conus404-hourly-T2').to_xarray()
    ds = ds.assign_coords(lat=dset['lat'], lon=dset['lon'])
    if 'crs' in ds:
        ds = ds.set_coords(['crs'])
    return ds


def compute_relative_humidity(ds: xr.Dataset) -> xr.DataArray:
    """
    Compute relative humidity from specific humidity, temperature, and pressure.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing 'Q2' (specific humidity), 'T2' (temperature in K), and 'PSFC' (pressure in Pa).

    Returns
    -------
    hurs : xr.DataArray
        Relative humidity as a percentage.
    """
    with xr.set_options(keep_attrs=True):
        hurs = convert.relative_humidity_from_dewpoint(tas=ds['T2'], tdps=ds['TD2'])
        hurs = cast(xr.DataArray, hurs)
    hurs.name = 'hurs'
    return hurs


def rotate_winds_to_earth(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Rotate grid-relative 10 m winds (U10,V10) to earth-relative components.
    Uses SINALPHA / COSALPHA convention from WRF.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing 'U10', 'V10', 'SINALPHA', and 'COSALPHA'.

    Returns
    -------
    earth_u : xr.DataArray
        Earth-relative U component of wind at 10 m.
    earth_v : xr.DataArray
        Earth-relative V component of wind at 10 m.


    """
    # rotate the grid-relative winds to earth-relative winds: https://forum.mmm.ucar.edu/threads/rotating-wrf-u-and-v-winds-before-and-after-reprojection.11788/
    # earth_u = u*cosa(ix,iy)-v*sina(ix,iy)
    # earth_v = v*cosa(ix,iy)+u*sina(ix,iy)
    with xr.set_options(keep_attrs=True):
        earth_u = ds.U10 * ds.COSALPHA - ds.V10 * ds.SINALPHA
        earth_v = ds.V10 * ds.COSALPHA + ds.U10 * ds.SINALPHA
    earth_u.name = 'u10_earth'
    earth_v.name = 'v10_earth'
    return earth_u, earth_v


def compute_wind_speed_and_direction(u10: xr.DataArray, v10: xr.DataArray) -> xr.Dataset:
    """Derive hourly wind speed (m/s) and direction (degrees from) using xclim.

    Parameters
    ----------
    u10 : xr.DataArray
        U component of wind at 10 m (m/s).
    v10 : xr.DataArray
        V component of wind at 10 m (m/s).

    Returns
    -------
    wind_ds : xr.Dataset
        Dataset containing wind speed ('sfcWind') and wind direction ('sfcWindfromdir').
    """
    winds = convert.wind_speed_from_vector(uas=u10, vas=v10)
    winds = cast(tuple[xr.DataArray, xr.DataArray], winds)
    # xclim returns a tuple-like (speed, direction). Merge keeps names (sfcWind, sfcWindfromdir)
    wind_ds = xr.merge(winds)
    return wind_ds
