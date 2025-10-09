import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Point

from ocr.testing import GeoDataFrameSnapshotExtension, XarraySnapshotExtension


@pytest.fixture
def xarray_snapshot(snapshot):
    return snapshot.use_extension(XarraySnapshotExtension)


@pytest.fixture
def geodataframe_snapshot(snapshot):
    return snapshot.use_extension(GeoDataFrameSnapshotExtension)


@pytest.mark.integration
def test_xarray_snapshot_extension(xarray_snapshot) -> None:
    data = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        dims=('x', 'y'),
    )
    assert xarray_snapshot == data


@pytest.mark.integration
def test_geodataframe_snapshot_extension(geodataframe_snapshot) -> None:
    """Test snapshot extension for GeoDataFrames."""
    gdf = gpd.GeoDataFrame(
        {
            'name': ['Point A', 'Point B', 'Point C'],
            'value': [10.5, 20.3, 30.7],
            'category': ['type1', 'type2', 'type1'],
        },
        geometry=[Point(-120.0, 35.0), Point(-119.5, 35.5), Point(-119.0, 36.0)],
        crs='EPSG:4326',
    )
    assert geodataframe_snapshot == gdf
