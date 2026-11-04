from typing import Any

import icechunk
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor
import zarr
from topozarr import create_pyramid


def write_multiscales(
    repo: icechunk.Repository,
    crs: str,
    *,
    branch: str = 'main',
    variables: list[str] | None = None,
    message: str = 'write multiscales',
    overwrite: bool = False,
    **pyramid_kwargs: Any,
) -> str:
    """Write multiscale levels 1..N over the base raster in group "0" and commit.

    `overwrite` first drops the existing levels (every group but "0"), so settings can change.
    Extra kwargs go to `topozarr.create_pyramid`. Returns the snapshot id.
    """
    session = repo.writable_session(branch)
    if overwrite:
        root = zarr.open_group(session.store, mode='r+')
        for name in [n for n in root.group_keys() if n != '0']:
            del root[name]
        root.attrs.pop('multiscales', None)
    ds = xr.open_zarr(session.store, group='0', consolidated=False, chunks=None)
    if variables is not None:
        ds = ds[variables]
    pyramid = create_pyramid(ds.proj.assign_crs(spatial_ref=crs), **pyramid_kwargs)
    pyramid.write(session.store, mode='a', levels=list(range(1, pyramid.levels)))
    return session.commit(message)
