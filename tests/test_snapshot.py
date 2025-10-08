import numpy as np
import pytest
import xarray as xr

from ocr.testing import XarraySnapshotExtension


@pytest.fixture
def xarray_snapshot(snapshot):
    return snapshot.use_extension(XarraySnapshotExtension)


@pytest.mark.integration
def test_xarray_snapshot_extension(xarray_snapshot) -> None:
    data = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        dims=('x', 'y'),
    )
    assert xarray_snapshot == data
