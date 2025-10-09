import os

import pytest
from upath import UPath

from ocr.config import OCRConfig
from ocr.pipeline.process_region import calculate_risk
from ocr.testing import GeoDataFrameSnapshotExtension, XarraySnapshotExtension
from ocr.types import RiskType

# Set default snapshot storage path to S3 if not already set
os.environ.setdefault('SNAPSHOT_STORAGE_PATH', 's3://carbonplan-ocr/integration-tests/snapshots/')


@pytest.fixture
def xarray_snapshot(snapshot):
    """Fixture for xarray snapshot testing.

    Available to all tests - use for snapshotting xarray DataArrays and Datasets.
    """
    return snapshot.use_extension(XarraySnapshotExtension)


@pytest.fixture
def geodataframe_snapshot(snapshot):
    """Fixture for GeoDataFrame snapshot testing.

    Available to all tests - use for snapshotting GeoPandas GeoDataFrames.
    """
    return snapshot.use_extension(GeoDataFrameSnapshotExtension)


@pytest.fixture(scope='session')
def ocr_config(tmp_path_factory):
    root = UPath(tmp_path_factory.mktemp('ocr_data'))
    cfg = OCRConfig(
        storage_root=str(root),
        debug=True,
        vector=None,
        icechunk=None,
        chunking=None,
        coiled=None,
    )
    cfg.icechunk.init_repo()
    return cfg


@pytest.fixture(scope='session')
def region_risk_parquet(ocr_config):
    # Single region used by multiple tests
    region_id = 'y2_x2'
    out_file = ocr_config.vector.region_geoparquet_uri / f'{region_id}.parquet'
    if not out_file.exists():
        calculate_risk(ocr_config, region_id=region_id, risk_type=RiskType.FIRE)
    return {
        'config': ocr_config,
        'region_id': region_id,
        'path': out_file,
    }
