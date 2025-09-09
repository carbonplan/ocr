import shutil
from typing import Any

import geopandas as gpd
import xarray as xr
from upath import UPath


def apply_s3_creds(region: str = 'us-west-2', *, con: Any | None = None):
    """Register AWS credentials as a DuckDB SECRET on the given connection.

    Parameters
    ----------
    region : str
        AWS region used for S3 access.
    con : duckdb.DuckDBPyConnection | None
        Connection to apply credentials to. If None, uses duckdb's default
        connection (duckdb.sql), preserving prior behavior.
    """
    import boto3
    import duckdb

    sess = boto3.Session()
    creds = sess.get_credentials()
    if creds is None:
        raise RuntimeError('No AWS credentials found by boto3.')
    frozen = creds.get_frozen_credentials()

    parts = [
        'CREATE OR REPLACE SECRET s3_default (',
        '  TYPE S3,',
        f"  KEY_ID '{frozen.access_key}',",
        f"  SECRET '{frozen.secret_key}',",
        f"  REGION '{region}'",
    ]
    if frozen.token:
        parts.append(f",  SESSION_TOKEN '{frozen.token}'")
    parts.append(');')
    sql = '\n'.join(parts)

    if con is None:
        duckdb.sql(sql)
    else:
        con.execute(sql)


def install_load_extensions(
    aws: bool = True, spatial: bool = True, httpfs: bool = True, con: Any | None = None
):
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
    con : duckdb.DuckDBPyConnection | None
        Connection to apply extensions to. If None, uses duckdb's default

    """
    import duckdb

    ext_str = ''
    if aws:
        ext_str += """INSTALL aws; LOAD aws;"""
    if spatial:
        ext_str += """INSTALL SPATIAL; LOAD SPATIAL;"""
    if httpfs:
        ext_str += """INSTALL httpfs; LOAD httpfs"""
    if con is None:
        duckdb.sql(ext_str)
    else:
        con.execute(ext_str)


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


def copy_or_upload(
    src: UPath, dest: UPath, overwrite: bool = True, chunk_size: int = 16 * 1024 * 1024
):
    """
    Copy a single file from src to dest using UPath/fsspec.
    - Uses server-side copy if available on the same filesystem (e.g., s3->s3).
    - Falls back to streaming copy otherwise.
    - Creates destination parent directories when supported.

    Parameters
    ----------

    src: UPath
        Source UPath
    dest: UPath
        Destination UPath (file path; if pointing to a directory-like path, src.name is appended)
    overwrite: bool
        If False, raises if dest exists
    chunk_size: int
        Buffer size for streaming copies

    Returns
    -------
    None
    """
    # If dest looks like a directory (exists as dir or endswith a separator), append filename
    if (dest.exists() and dest.is_dir()) or str(dest).endswith(('/', '\\')):
        dest = dest / src.name

    # Try to ensure destination parent exists (no-op for object stores)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # If both paths are on the same filesystem and it supports copy, do a server-side copy
    try:
        same_fs = type(src.fs) is type(dest.fs)
        if same_fs and hasattr(src.fs, 'copy'):
            # Some fs implementations accept overwrite/recursive; keep it simple and let overwrite control existence
            if not overwrite and dest.exists():
                raise FileExistsError(f'Destination already exists: {dest}')
            src.fs.copy(str(src), str(dest))
            return
    except Exception:
        # Fall back to streaming if server-side copy fails for any reason
        pass

    # Streaming copy between filesystems (or when server-side copy isn't available)
    mode = 'wb' if overwrite else 'xb'
    with src.open('rb') as r, dest.open(mode) as w:
        shutil.copyfileobj(r, w, length=chunk_size)
