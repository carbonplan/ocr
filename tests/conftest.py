import os

import pytest
from odc.geo.xr import assign_crs
from upath import UPath

from ocr import catalog
from ocr.config import OCRConfig
from ocr.pipeline.process_region import calculate_risk
from ocr.risks.fire import calculate_wind_adjusted_risk, create_wind_informed_burn_probability
from ocr.testing import GeoDataFrameSnapshotExtension, XarraySnapshotExtension
from ocr.types import RiskType
from ocr.utils import geo_sel

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
    'california-coast': (slice(-120.0, -119.985), slice(35.005, 35.0)),
    'colorado-rockies': (slice(-105.0, -104.985), slice(40.005, 40.0)),
    'seattle-area': (slice(-122.5, -122.485), slice(47.605, 47.6)),
    'georgia-piedmont': (slice(-84.4, -84.385), slice(33.755, 33.75)),
    'arizona-desert': (slice(-111.9, -111.885), slice(33.455, 33.45)),
}


@pytest.fixture(scope='session')
def wind_informed_bp_cache():
    """Cache for expensive wind-informed burn probability calculations.

    Session-scoped fixture that stores results to avoid recomputing the same
    regions across different test modules. Returns a dict-like cache object.
    """
    return {}


@pytest.fixture(scope='session')
def get_wind_informed_burn_probability(wind_informed_bp_cache):
    """Factory fixture to get or compute wind-informed burn probability for a region.

    This fixture returns a function that:
    1. Checks if the result is already cached
    2. If not, computes it and caches it
    3. Returns the cached result

    Usage in tests:
        def test_something(get_wind_informed_burn_probability):
            bp_2011 = get_wind_informed_burn_probability('california-coast')
    """

    def _get_bp(region_id, x_slice=None, y_slice=None):
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
        if cache_key not in wind_informed_bp_cache:
            # Load the required data
            riley_2011_270m_5070 = catalog.get_dataset('2011-climate-run').to_xarray()[
                ['BP', 'spatial_ref']
            ]
            riley_2011_270m_5070 = assign_crs(riley_2011_270m_5070, 'EPSG:5070')

            riley_2011_270m_5070_subset = geo_sel(
                riley_2011_270m_5070,
                bbox=(x_slice.start, y_slice.stop, x_slice.stop, y_slice.start),
                crs_wkt=riley_2011_270m_5070.spatial_ref.attrs['crs_wkt'],
            )

            wind_direction_distribution_30m_4326 = (
                catalog.get_dataset('conus404-ffwi-p99-wind-direction-distribution-reprojected')
                .to_xarray()
                .wind_direction_distribution.sel(latitude=y_slice, longitude=x_slice)
                .load()
            )

            # Compute and cache
            wind_informed_bp_cache[cache_key] = create_wind_informed_burn_probability(
                wind_direction_distribution_30m_4326=wind_direction_distribution_30m_4326,
                riley_270m_5070=riley_2011_270m_5070_subset,
            )

        return wind_informed_bp_cache[cache_key]

    return _get_bp


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
