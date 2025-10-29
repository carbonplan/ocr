import os

import pytest
from odc.geo.xr import assign_crs

from ocr import catalog
from ocr.risks.fire import calculate_wind_adjusted_risk, create_wind_informed_burn_probability
from ocr.testing import GeoDataFrameSnapshotExtension, XarraySnapshotExtension
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
    'california-coast': (slice(-120.0, -119.985), slice(35.0, 35.005)),
    'colorado-rockies': (slice(-105.0, -104.985), slice(40.0, 40.005)),
    'seattle-area': (slice(-122.5, -122.485), slice(47.6, 47.605)),
    'georgia-piedmont': (slice(-84.4, -84.385), slice(33.75, 33.755)),
    'arizona-desert': (slice(-111.9, -111.885), slice(33.45, 33.455)),
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
            # Apply buffer to match calculate_wind_adjusted_risk behavior
            buffer = 0.15
            buffered_x_slice = slice(x_slice.start - buffer, x_slice.stop + buffer, x_slice.step)
            buffered_y_slice = slice(y_slice.start - buffer, y_slice.stop + buffer, y_slice.step)

            # Load the required data with buffered slices
            riley_2011_270m_5070 = catalog.get_dataset(
                'riley-et-al-2025-2011-270m-5070'
            ).to_xarray()[['BP', 'spatial_ref']]
            riley_2011_270m_5070 = assign_crs(riley_2011_270m_5070, 'EPSG:5070')

            # west, south, east, north = bbox
            bbox = (
                buffered_x_slice.start,
                buffered_y_slice.start,
                buffered_x_slice.stop,
                buffered_y_slice.stop,
            )

            riley_2011_270m_5070_subset = geo_sel(
                riley_2011_270m_5070,
                bbox=bbox,
                crs_wkt=riley_2011_270m_5070.spatial_ref.attrs['crs_wkt'],
            )

            wind_direction_distribution_30m_4326 = (
                catalog.get_dataset('conus404-ffwi-p99-wind-direction-distribution-30m-4326')
                .to_xarray()
                .wind_direction_distribution.sel(
                    latitude=buffered_y_slice, longitude=buffered_x_slice
                )
                .load()
            )

            # Compute wind-informed BP with buffered data
            wind_informed_bp_full = create_wind_informed_burn_probability(
                wind_direction_distribution_30m_4326=wind_direction_distribution_30m_4326,
                riley_270m_5070=riley_2011_270m_5070_subset,
            ).compute()

            # Clip to original (non-buffered) extent
            wind_informed_bp_cache[cache_key] = wind_informed_bp_full.sel(
                latitude=y_slice, longitude=x_slice
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
            ).compute()

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
