import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyproj
import xarray as xr
from upath import UPath


def get_temp_dir() -> Path | None:
    """Get optimal temporary directory path for the current environment.

    Returns the current working directory if running in /scratch (e.g., on Coiled
    clusters), otherwise returns None to use the system default temp directory.

    On Coiled clusters, /scratch is bind-mounted directly to the NVMe disk,
    avoiding Docker overlay filesystem overhead and providing better I/O performance
    and more available space compared to /tmp which sits on the Docker overlay.

    Returns
    -------
    Path | None
        Current working directory if in /scratch, None otherwise (uses system default).

    Examples
    --------
    >>> import tempfile
    >>> from ocr.utils import get_temp_dir
    >>> with tempfile.TemporaryDirectory(dir=get_temp_dir()) as tmpdir:
    ...     # tmpdir will be in /scratch on Coiled, system temp otherwise
    ...     pass
    """
    cwd = Path.cwd()
    if cwd.parts[1:2] == ('scratch',):
        # Return /scratch itself rather than a subdirectory within it
        return Path('/scratch')
    return None


def apply_s3_creds(region: str = 'us-west-2', *, con: Any | None = None) -> None:
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
) -> None:
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


def bbox_tuple_from_xarray_extent(
    ds: xr.Dataset, x_name: str = 'x', y_name: str = 'y'
) -> tuple[float, float, float, float]:
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
) -> None:
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


@lru_cache
def _get_transformers(wkt: str):
    crs_proj = pyproj.CRS.from_wkt(wkt)
    crs_geo = pyproj.CRS.from_epsg(4326)
    fwd = pyproj.Transformer.from_crs(crs_geo, crs_proj, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs_proj, crs_geo, always_xy=True)
    return fwd, inv


def geo_sel(
    ds: xr.Dataset,
    *,
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
