from typing import Any

import icechunk
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor
from topozarr import create_pyramid


def write_multiscales(
    repo: icechunk.Repository,
    crs: str,
    *,
    branch: str = 'main',
    variables: list[str] | None = None,
    message: str = 'write multiscales',
    **pyramid_kwargs: Any,
) -> str:
    """Write multiscale levels 1..N over the base raster in group "0" and commit.

    Extra kwargs go to `topozarr.create_pyramid`. Returns the snapshot id.
    """
    session = repo.writable_session(branch)
    ds = xr.open_zarr(session.store, group='0', consolidated=False, chunks=None)
    if variables is not None:
        ds = ds[variables]
    pyramid = create_pyramid(ds.proj.assign_crs(spatial_ref=crs), **pyramid_kwargs)
    pyramid.write(session.store, mode='a', levels=list(range(1, pyramid.levels)))
    return session.commit(message)
