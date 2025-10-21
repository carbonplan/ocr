from functools import lru_cache
from typing import cast

import pyproj
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


@lru_cache
def _get_transformers(wkt: str):
    crs_proj = pyproj.CRS.from_wkt(wkt)
    crs_geo = pyproj.CRS.from_epsg(4326)
    fwd = pyproj.Transformer.from_crs(crs_geo, crs_proj, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs_proj, crs_geo, always_xy=True)
    return fwd, inv


def geo_sel(
    ds: xr.Dataset,
    lon: float | None = None,
    lat: float | None = None,
    bbox: tuple[float, float, float, float] | None = None,  # (west, south, east, north)
    method: str = 'nearest',
    tolerance: float | None = None,
    crs_wkt: str | None = None,
):
    """
    Geographic selection helper.

    Exactly one of:
      - (lon AND lat)
      - (lons AND lats)
      - bbox=(west, south, east, north)

    Parameters
    ----------
    ds : xr.Dataset
      Input dataset with x, y coordinates and a valid 'crs' variable with WKT
    lon : float, optional
      Longitude of point to select, by default None
    lat : float, optional
      Latitude of point to select, by default None
    bbox : tuple, optional
      Bounding box to select (west, south, east, north), by default None
    method : str, optional
      Method to use for point selection, by default 'nearest'
    tolerance : float, optional
      Tolerance (in units of the dataset's CRS) for point selection, by default None
    crs_wkt : str, optional
      WKT string for the dataset's CRS. If None, attempts to read from ds.crs.attrs['crs_wkt'].

    Returns
    -------
    xarray.Dataset
      Single point: time dimension only
      Multiple points: adds 'point' dimension
      BBox: retains y, x subset
    """
    if crs_wkt is None:
        try:
            wkt = ds.crs.attrs['crs_wkt']
        except KeyError:
            raise ValueError(
                'CRS WKT not found in dataset attributes. Please provide crs_wkt argument.'
            )
    else:
        wkt = crs_wkt
    fwd, _ = _get_transformers(wkt)

    # --- Case 1: Bounding box ---
    if bbox is not None:
        if any(v is not None for v in (lon, lat)):
            raise ValueError('Provide either bbox OR point(s), not both.')
        west, south, east, north = bbox
        # Project the 4 corners
        xs, ys = zip(
            *[
                fwd.transform(west, south),
                fwd.transform(east, south),
                fwd.transform(east, north),
                fwd.transform(west, north),
            ]
        )
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Handle coordinate order (assumes monotonic x and y)
        x_asc = ds.x[0] < ds.x[-1]
        y_asc = ds.y[0] < ds.y[-1]

        x_slice = slice(x_min, x_max) if x_asc else slice(x_max, x_min)
        y_slice = slice(y_min, y_max) if y_asc else slice(y_max, y_min)

        return ds.sel(x=x_slice, y=y_slice)

    # --- Case 2: Single point ---
    if lon is not None and lat is not None:
        x_pt, y_pt = fwd.transform(lon, lat)
        out = ds.sel(x=x_pt, y=y_pt, method=method, tolerance=tolerance)
        # Optional: add requested lon/lat as attributes
        out.attrs['requested_lon'] = float(lon)
        out.attrs['requested_lat'] = float(lat)
        return out

    raise ValueError('You must supply either (lon & lat), (lons & lats), or bbox.')
