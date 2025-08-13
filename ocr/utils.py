from __future__ import annotations

import typing

import geopandas as gpd
import icechunk
import xarray as xr

from ocr.console import console

if typing.TYPE_CHECKING:
    import xarray as xr


def get_commit_messages_ancestry(repo: icechunk.Repository, *, branch: str = 'main') -> list:
    commit_messages = [commit.message for commit in list(repo.ancestry(branch=branch))]
    # separate commits by ',' and handle case of single length ancestry commit history
    split_commits = [
        msg
        for message in commit_messages
        for msg in (message.split(',') if ',' in message else [message])
    ]
    return split_commits


def insert_region_uncoop(
    session: icechunk.Session,
    *,
    subset_ds: xr.Dataset,
    region_id: str,
):
    import icechunk

    console.log(f'Inserting region: {region_id} into Icechunk store: ')

    while True:
        try:
            subset_ds.to_zarr(
                session.store,
                region='auto',
                consolidated=False,
            )
            # Trying out the rebase strategy described here: https://github.com/earth-mover/icechunk/discussions/802#discussioncomment-13064039
            # We should be in the same position, where we don't have real conflicts, just write timing conflicts.
            session.commit(f'{region_id}', rebase_with=icechunk.ConflictDetector())
            console.log(f'Wrote dataset: {subset_ds} to region: {region_id}')
            break

        except icechunk.ConflictError:
            console.log(f'conflict for region_commit_history {region_id}, retrying')
            pass


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
