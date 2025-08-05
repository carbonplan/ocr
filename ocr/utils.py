import geopandas as gpd
import icechunk
import xarray as xr
from zarr.codecs import BloscCodec


def apply_s3_creds(region: str = 'us-west-2') -> None:
    """
    Applies duckdb region and access credentials to session.

    Parameters
    ----------
    region : str, optional
        AWS Region, by default 'us-west-2'

    """
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
    """
    Installs and applies duckdb extensions.

    Parameters
    ----------
    aws : bool, optional
        Install and load AWS extension, by default True
    spatial : bool, optional
        Install and load SPATIAL extension, by default True
    httpfs : bool, optional
        Install and load HTTPFS extension, by default True

    """
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
    """
    Convert longitude values from 0-360 to -180-180.

    Note: `longitude` is required dim/coord.

    Parameters
    ----------
    ds : xr.Dataset
        Input Xarray dataset

    Returns
    -------
    xr.Dataset
        Dataset with longitude coordinates converted to -180-180 range
    """
    lon = ds['longitude'].where(ds['longitude'] < 180, ds['longitude'] - 360)
    ds = ds.assign_coords(longitude=lon)
    return ds


# TODO: unused DEPRECIATE?

# def subset_region_latlon(ds: xr.Dataset, lon_range: Iterable, lat_range: Iterable) -> xr.Dataset:
#     """Subset an Xarray dataset by a lat / lon range.

#     Parameters
#     ----------
#     ds : xr.Dataset
#         Input Xarray dataset
#     lon_range : Iterable
#         Longitude range as [min_lon, max_lon]
#     lat_range : Iterable
#         Latitude range as [min_lat, max_lat]

#     Returns
#     -------
#     xr.Dataset
#         Dataset subset to the specified lat/lon range
#     """
#     import geopandas as gpd

#     points = gpd.points_from_xy(lon_range, lat_range, crs='EPSG:4326')
#     points = points.to_crs('EPSG:5070')
#     region = ds.sel(x=slice(points.x[0], points.x[1]), y=slice(points.y[1], points.y[0]))
#     return region

# TODO: unused DEPRECIATE?
# def subset_region_xy(ds: xr.Dataset, x_range, y_range) -> xr.Dataset:
#     """
#     Subset an Xarray dataset by x/y coordinate range.

#     Parameters
#     ----------
#     ds : xr.Dataset
#         Input Xarray dataset
#     x_range : tuple or list
#         X coordinate range as [min_x, max_x]
#     y_range : tuple or list
#         Y coordinate range as [min_y, max_y]

#     Returns
#     -------
#     xr.Dataset
#         Dataset subset to the specified x/y range
#     """
#     region = ds.sel(x=slice(x_range[0], x_range[1]), y=slice(y_range[1], y_range[0]))
#     return region


# TODO: single line function DEPRECIATE?
# def interpolate_to_30(da, target):
#     """
#     Interpolate DataArray to match target coordinates.

#     Parameters
#     ----------
#     da : xr.DataArray
#         Input DataArray to interpolate
#     target : xr.DataArray
#         Target DataArray with desired coordinates

#     Returns
#     -------
#     xr.DataArray
#         Interpolated DataArray

#     Notes
#     -----
#     TODO - prevent the interpolation from making negative risk values
#     """
#     return da.interp_like(target, kwargs={'fill_value': 'extrapolate', 'bounds_error': False})


def convert_coords(
    coords: list[tuple[float, float]] | gpd.GeoDataFrame, from_crs: str, to_crs: str
) -> list[tuple[float, float]] | gpd.GeoDataFrame:
    """
    Convert coordinates between xy and latlon using GeoPandas.

    Parameters
    ----------
    coords : list of tuple or gpd.GeoDataFrame
        The coordinates to convert. Can be a list of (x, y) or (lon, lat) tuples,
        or a GeoDataFrame with a geometry column
    from_crs : str
        The CRS of the input coordinates (e.g., "EPSG:4326" for latlon or "EPSG:5070" for xy)
    to_crs : str
        The CRS to convert the coordinates to (e.g., "EPSG:4326" for latlon or "EPSG:5070" for xy)

    Returns
    -------
    list of tuple or gpd.GeoDataFrame
        The converted coordinates in the target CRS

    Raises
    ------
    ValueError
        If input GeoDataFrame does not have a CRS defined
    TypeError
        If input is not a list of tuples or a GeoDataFrame
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
    """
    Extract/sample points from a GeoDataFrame to an Xarray DataArray.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geopandas GeoDataFrame. Geometry should be points
    da : xr.DataArray
        Input Xarray DataArray

    Returns
    -------
    xr.DataArray
        DataArray with geometry sampled

    Notes
    -----
    UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are
    likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a
    projected CRS before this operation.

    The relatively small size of a building footprint should account for a very
    small shift in the centroid when calculating from EPSG:4326 vs EPSG:5070.

    TODO: Should/can this be a DataArray for typing
    """
    import xarray as xr

    x_coords, y_coords = gdf.geometry.centroid.x, gdf.geometry.centroid.y

    nearest_pixels = da.sel(
        longitude=xr.DataArray(x_coords, dims='points'),
        latitude=xr.DataArray(y_coords, dims='points'),
        method='nearest',
    )
    return nearest_pixels.values


def bbox_tuple_from_xarray_extent(ds: xr.Dataset, x_name: str = 'x', y_name: str = 'y') -> tuple:
    """
    Creates a bounding box from an Xarray Dataset extent.

    Parameters
    ----------
    ds : xr.Dataset
        Input Xarray Dataset
    x_name : str, optional
        Name of x coordinate, by default 'x'
    y_name : str, optional
        Name of y coordinate, by default 'y'

    Returns
    -------
    tuple
        Bounding box tuple in the form: (x_min, y_min, x_max, y_max)
    """
    x_min = float(ds[x_name].min())
    x_max = float(ds[x_name].max())
    y_min = float(ds[y_name].min())
    y_max = float(ds[y_name].max())
    return (x_min, y_min, x_max, y_max)


def prep_encoding(ds):
    var_list = list(ds.keys())
    encoding = {}
    for var in var_list:
        encoding[var] = {
            'compressor': BloscCodec(
                cname='zstd',
                clevel=6,
            )
        }
    for coord in ds.coords:
        encoding[coord] = {'compressor': None}
    return encoding


def load_conus404(variable):
    config = {
        'bucket': 'carbonplan-ocr',
        'prefix': f'input/conus404-hourly-icechunk/{variable}',
        'region': 'us-west-2',
    }

    storage = icechunk.s3_storage(
        bucket=config['bucket'],
        prefix=config['prefix'],
        region=config['region'],
    )

    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session('main')
    return xr.open_zarr(session.store, consolidated=False)
