from collections.abc import Iterable

import geopandas as gpd
import xarray as xr


def apply_s3_creds(region: str = 'us-west-2'):
    import boto3
    import duckdb

    session = boto3.Session()
    credentials = session.get_credentials()
    return duckdb.sql(f"""CREATE SECRET (
    TYPE s3,
    KEY_ID '{credentials.access_key}',
    SECRET '{credentials.secret_key}',
    REGION '{region}');""")


def install_load_extensions(aws: bool = True, spatial: bool = True, httpfs: bool = True):
    import duckdb

    ext_str = ''
    if aws:
        ext_str += """INSTALL aws; LOAD aws;"""
    if spatial:
        ext_str += """INSTALL SPATIAL; LOAD SPATIAL;"""
    if httpfs:
        ext_str += """INSTALL httpfs; LOAD httpfs"""
    return duckdb.sql(ext_str)


def lon_to_180(ds: xr.Dataset) -> xr.Dataset:
    lon = ds['longitude'].where(ds['longitude'] < 180, ds['longitude'] - 360)
    ds = ds.assign_coords(longitude=lon)
    return ds


def subset_region_latlon(ds: xr.Dataset, lon_range: Iterable, lat_range: Iterable) -> xr.Dataset:
    import geopandas as gpd

    points = gpd.points_from_xy(lon_range, lat_range, crs='EPSG:4326')
    points = points.to_crs('EPSG:5070')
    region = ds.sel(x=slice(points.x[0], points.x[1]), y=slice(points.y[1], points.y[0]))
    return region


def subset_region_xy(ds: xr.Dataset, x_range, y_range) -> xr.Dataset:
    region = ds.sel(x=slice(x_range[0], x_range[1]), y=slice(y_range[1], y_range[0]))
    return region


def interpolate_to_30(da, target):
    # TODO - prevent the interpolation from making negative risk values
    return da.interp_like(target, kwargs={'fill_value': 'extrapolate', 'bounds_error': False})


def convert_coords(
    coords: list[tuple[float, float]] | gpd.GeoDataFrame, from_crs: str, to_crs: str
) -> list[tuple[float, float]] | gpd.GeoDataFrame:
    """
    Convert coordinates between xy and latlon using GeoPandas.

    Parameters:
    - coords: list of tuples or GeoDataFrame
        The coordinates to convert. Can be a list of (x, y) or (lon, lat) tuples,
        or a GeoDataFrame with a geometry column.
    - from_crs: str
        The CRS of the input coordinates (e.g., "EPSG:4326" for latlon or "EPSG:5070" for xy).
    - to_crs: str
        The CRS to convert the coordinates to (e.g., "EPSG:4326" for latlon or "EPSG:5070" for xy).

    Returns:
    - converted_coords: list of tuples or GeoDataFrame
        The converted coordinates in the target CRS.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    # If input is a list of tuples, create a GeoDataFrame
    if isinstance(coords, list):
        gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coords], crs=from_crs)
    elif isinstance(coords, gpd.GeoDataFrame):
        gdf = coords
        if gdf.crs is None:
            raise ValueError('Input GeoDataFrame must have a CRS defined.')
    else:
        raise TypeError('Input must be a list of tuples or a GeoDataFrame.')

    # Convert to the target CRS
    gdf_converted = gdf.to_crs(to_crs)

    # If input was a list, return the converted coordinates as a list of tuples
    if isinstance(coords, list):
        return [(geom.x, geom.y) for geom in gdf_converted.geometry]

    # If input was a GeoDataFrame, return the converted GeoDataFrame
    return gdf_converted


def extract_points(gdf: gpd.GeoDataFrame, da: xr.DataArray) -> xr.DataArray:
    import xarray as xr

    # ensure CRS alignment
    if gdf.crs != da.rio.crs:
        da = da.rio.reproject(gdf.crs)

    x_coords, y_coords = gdf.geometry.centroid.x, gdf.geometry.centroid.y

    nearest_pixels = da.sel(
        x=xr.DataArray(x_coords, dims='points'),
        y=xr.DataArray(y_coords, dims='points'),
        method='nearest',
    )

    return nearest_pixels.values


def bbox_tuple_from_xarray_extent(ds: xr.Dataset, x_name: str = 'x', y_name: str = 'y') -> tuple:
    x_min = float(ds[x_name].min())
    x_max = float(ds[x_name].max())
    y_min = float(ds[y_name].min())
    y_max = float(ds[y_name].max())
    return (x_min, y_min, x_max, y_max)
