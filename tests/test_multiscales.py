import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr

from ocr.multiscales import write_multiscales


def _repo_with_base(n: int) -> icechunk.Repository:
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session('main')
    ds = xr.Dataset(
        {'risk': (('y', 'x'), np.random.rand(n, n).astype('f4'))},
        coords={'y': np.arange(n, dtype='f8'), 'x': np.arange(n, dtype='f8')},
    )
    ds.to_zarr(session.store, group='0', zarr_format=3, consolidated=False)
    session.commit('base')
    return repo


def test_write_multiscales() -> None:
    repo = _repo_with_base(1024)
    snap = write_multiscales(repo, 'EPSG:4326', levels=3)
    assert snap == repo.lookup_branch('main')
    root = zarr.open_group(repo.readonly_session('main').store, mode='r')
    assert root['1/risk'].shape == (512, 512)
    assert root['2/risk'].shape == (256, 256)
    assert len(root.attrs['multiscales']['layout']) == 3


def test_write_multiscales_rerun() -> None:
    repo = _repo_with_base(1024)
    write_multiscales(repo, 'EPSG:4326', levels=3)
    write_multiscales(repo, 'EPSG:4326', levels=3)
    head = repo.lookup_branch('main')
    with pytest.raises(ValueError, match='does not match this pyramid'):
        write_multiscales(repo, 'EPSG:4326', levels=2)
    assert repo.lookup_branch('main') == head


def test_write_multiscales_too_small() -> None:
    repo = _repo_with_base(300)
    head = repo.lookup_branch('main')
    with pytest.raises(ValueError, match='no levels to write'):
        write_multiscales(repo, 'EPSG:4326', levels='auto')
    assert repo.lookup_branch('main') == head
