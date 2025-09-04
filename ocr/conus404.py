from functools import lru_cache
from typing import cast

import pyproj
import xarray as xr
import xclim.indicators.atmos
from odc.geo import CRS
from odc.geo.xr import assign_crs

from ocr import catalog


def load_conus404(add_spatial_constants: bool = True) -> xr.Dataset:
    """
    Load the CONUS 404 dataset.

    Parameters
    ----------
    add_spatial_constants : bool, optional
        If True, adds spatial constant variables (SINALPHA, COSALPHA) to the dataset.

    Returns
    -------
    xr.Dataset
        The CONUS 404 dataset.
    """

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
    xr.DataArray
        Relative humidity as a percentage.
    """
    with xr.set_options(keep_attrs=True):
        hurs = xclim.indicators.atmos.relative_humidity_from_dewpoint(tas=ds['T2'], tdps=ds['TD2'])
    hurs.name = 'hurs'
    return hurs


def rotate_winds_to_earth(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Rotate grid-relative 10 m winds (U10,V10) to earth-relative components.

    Uses SINALPHA / COSALPHA convention from WRF (same as notebook).

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
    """Derive hourly wind speed (m/s) and direction (degrees from) using xclim."""
    winds = xclim.indicators.atmos.wind_speed_from_vector(uas=u10, vas=v10)
    # xclim returns a tuple-like (speed, direction). Merge keeps names (sfcWind, sfcWindfromdir)
    wind_ds = xr.merge(winds)
    return wind_ds


def build_fire_weather_mask(
    hurs: xr.DataArray,
    wind_ds: xr.Dataset,
    *,
    hurs_threshold: float,
    wind_threshold: float,
    wind_gust_factor: float = 1.4,
) -> xr.DataArray:
    """Compute a boolean fire weather mask.

    Applies gust factor to sustained wind speed before thresholding.
    """
    from ocr.risks.fire import nws_fire_weather

    # Convert sustained to approximate gusts via multiplier
    # reason that wind gusts are typically ~40% higher than average wind speed
    # and we want to base this on wind gusts (need a citation for this)
    gust_like = wind_ds['sfcWind'] * wind_gust_factor
    mask = nws_fire_weather(
        hurs, hurs_threshold, gust_like, wind_threshold, tas=None, tas_threshold=None
    )
    mask.name = 'fire_weather_mask'
    return mask


def compute_modal_wind_direction(
    direction: xr.DataArray,
    fire_weather_mask: xr.DataArray,
) -> xr.Dataset:
    """Compute modal wind direction (0-7) for hours satisfying fire weather.

    Direction codes follow: 0=N,1=NE,2=E,3=SE,4=S,5=SW,6=W,7=NW
    """
    from ocr.risks.fire import classify_wind_directions, direction_histogram

    direction_indices = classify_wind_directions(direction)
    masked = direction_indices.where(fire_weather_mask)
    fraction = direction_histogram(masked)
    # Help static type checkers – ensure fraction is treated as DataArray
    fraction = cast(xr.DataArray, fraction)
    assert isinstance(fraction, xr.DataArray)

    # Identify pixels with any fire-weather hours (probabilities sum to 1 else 0)
    any_fire_weather = fraction.sum(dim='wind_direction') > 0

    mode = fraction.argmax(dim='wind_direction').where(any_fire_weather).chunk({'x': -1, 'y': -1})
    mode.name = 'wind_direction_mode'
    mode.attrs.update(
        {
            'long_name': 'Modal wind direction during fire-weather hours',
            'description': 'Most frequent of 8 cardinal directions during hours meeting fire weather criteria',
            'direction_labels': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
            'fire_weather_definition': 'RH < hurs_threshold (%), gust_like_wind > wind_threshold (mph)',
        }
    )
    return mode.to_dataset()


def reproject_mode(
    mode: xr.Dataset,
    src_crs_wkt: str,
    target_dataset_name: str,
    *,
    chunk_lat: int = 6000,
    chunk_lon: int = 4500,
) -> xr.Dataset:
    """Reproject the modal wind direction to the geobox of a target dataset."""
    tgt = catalog.get_dataset(target_dataset_name).to_xarray().astype('float32')
    tgt = assign_crs(tgt, crs='EPSG:4326')
    geobox = tgt.odc.geobox

    src_crs = CRS(src_crs_wkt)
    mode_src = assign_crs(mode, crs=src_crs)

    result = (
        mode_src.odc.reproject(geobox, resampling='nearest')
        .astype('float32')
        .chunk({'latitude': chunk_lat, 'longitude': chunk_lon})
    )

    result.attrs.update({'reprojected_to': target_dataset_name})
    return result


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
):
    """
    Geographic selection helper.

    Exactly one of:
      - (lon AND lat)
      - (lons AND lats)
      - bbox=(west, south, east, north)

    Returns
    -------
    xarray.Dataset
      Single point: time dimension only
      Multiple points: adds 'point' dimension
      BBox: retains y, x subset
    """
    wkt = ds.crs.attrs['crs_wkt']
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
