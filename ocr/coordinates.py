import geopandas as gpd
import xarray as xr
from shapely.geometry import Point


def lon_to_180(ds: xr.Dataset) -> xr.Dataset:
    lon = ds['longitude'].where(ds['longitude'] < 180, ds['longitude'] - 360)
    ds = ds.assign_coords(longitude=lon)
    return ds


def convert_coords(coords, from_crs: str, to_crs: str) -> list | gpd.GeoDataFrame:
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
    if gdf.crs != da.rio.crs:
        da = da.rio.reproject(gdf.crs)

    x_coords, y_coords = gdf.geometry.centroid.x, gdf.geometry.centroid.y

    nearest_pixels = da.sel(
        x=xr.DataArray(x_coords, dims='points'),
        y=xr.DataArray(y_coords, dims='points'),
        method='nearest',
    )

    return nearest_pixels.values
