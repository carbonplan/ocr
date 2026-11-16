from typing import Any

import icechunk
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor
import zarr
from topozarr import create_pyramid

# icechunk<=2.3.1 str(store) waits on the session lock holding the GIL; zarr>=3.4.1 calls it: hang
icechunk.IcechunkStore.__str__ = lambda self: 'icechunk.IcechunkStore'


def drop_levels(session: icechunk.Session) -> None:
    """Delete the pyramid levels (every group but "0") and the root multiscales attrs; uncommitted."""
    root = zarr.open_group(session.store, mode='r+')
    for name in [n for n in root.group_keys() if n != '0']:
        del root[name]
    root.attrs.pop('multiscales', None)


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
        drop_levels(session)
    ds = xr.open_zarr(session.store, group='0', consolidated=False, chunks=None)
    if variables is not None:
        ds = ds[variables]
    pyramid = create_pyramid(ds.proj.assign_crs(spatial_ref=crs), **pyramid_kwargs)
    pyramid.write(session.store, mode='a', levels=list(range(1, pyramid.levels)))
    return session.commit(message)
