import os

import pytest
from upath import UPath

from ocr.config import OCRConfig
from ocr.pipeline.process_region import calculate_risk
from ocr.risks.fire import calculate_wind_adjusted_risk
from ocr.testing import GeoDataFrameSnapshotExtension, XarraySnapshotExtension
from ocr.types import RiskType

# Set default snapshot storage path to S3 if not already set
os.environ.setdefault('SNAPSHOT_STORAGE_PATH', 's3://carbonplan-ocr/integration-tests/snapshots/')


@pytest.fixture(scope='session', autouse=True)
def cleanup_s3_directory():
    """Clean up incorrectly created 's3:' directory from syrupy."""
    yield
    # Cleanup after all tests complete
    import pathlib
    import shutil

    s3_dir = pathlib.Path('s3:')
    if s3_dir.exists():
        shutil.rmtree(s3_dir, ignore_errors=True)


# Define test regions used across multiple test modules
TEST_REGIONS = {
    'california-coast': (slice(-120.0, -119.995), slice(35.005, 35.0)),
    'colorado-rockies': (slice(-105.0, -104.995), slice(40.005, 40.0)),
    'seattle-area': (slice(-122.5, -122.495), slice(47.605, 47.6)),
    'georgia-piedmont': (slice(-84.4, -84.395), slice(33.755, 33.75)),
    'arizona-desert': (slice(-111.9, -111.895), slice(33.455, 33.45)),
}


@pytest.fixture(scope='session')
def wind_adjusted_risk_cache():
    """Cache for expensive wind-adjusted risk calculations.

    Session-scoped fixture that stores results to avoid recomputing the same
    regions across different test modules. Returns a dict-like cache object.
    """
    return {}


@pytest.fixture(scope='session')
def get_wind_adjusted_risk(wind_adjusted_risk_cache):
    """Factory fixture to get or compute wind-adjusted risk for a region.

    This fixture returns a function that:
    1. Checks if the result is already cached
    2. If not, computes it and caches it
    3. Returns the cached result

    Usage in tests:
        def test_something(get_wind_adjusted_risk):
            risk_ds = get_wind_adjusted_risk('california_coast')
            # or with custom coords:
            risk_ds = get_wind_adjusted_risk(
                'custom_region',
                x_slice=slice(-120, -119.9),
                y_slice=slice(35, 34.9)
            )
    """

    def _get_risk(region_id, x_slice=None, y_slice=None):
        # Use region_id to look up coords if not provided
        if x_slice is None or y_slice is None:
            if region_id not in TEST_REGIONS:
                raise ValueError(
                    f"Unknown region '{region_id}'. "
                    f'Available: {list(TEST_REGIONS.keys())} '
                    'or provide x_slice and y_slice explicitly.'
                )
            x_slice, y_slice = TEST_REGIONS[region_id]

        # Check cache
        cache_key = f'{region_id}_{x_slice}_{y_slice}'
        if cache_key not in wind_adjusted_risk_cache:
            # Compute and cache
            wind_adjusted_risk_cache[cache_key] = calculate_wind_adjusted_risk(
                x_slice=x_slice, y_slice=y_slice
            )

        return wind_adjusted_risk_cache[cache_key]

    return _get_risk


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
