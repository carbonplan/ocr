from collections.abc import Iterable

import geopandas as gpd
import xarray as xr


def subset_region_latlon(ds: xr.Dataset, lon_range: Iterable, lat_range: Iterable) -> xr.Dataset:
    points = gpd.points_from_xy(lon_range, lat_range, crs='EPSG:4326')
    points = points.to_crs('EPSG:5070')
    region = ds.sel(x=slice(points.x[0], points.x[1]), y=slice(points.y[1], points.y[0]))
    return region


def subset_region_xy(ds: xr.Dataset, x_range, y_range) -> xr.Dataset:
    region = ds.sel(x=slice(x_range[0], x_range[1]), y=slice(y_range[1], y_range[0]))
    return region
