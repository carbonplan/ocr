import pytest

from ocr.pipeline.process_region import sample_risk_to_buildings
from ocr.risks.fire import calculate_wind_adjusted_risk

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    'region_coords',
    [
        pytest.param((slice(-120.0, -119.9), slice(35.1, 35.0)), id='california_coast'),
        pytest.param((slice(-105.0, -104.9), slice(40.1, 40.0)), id='rocky_mountains'),
    ],
)
def test_sample_risk_to_buildings_snapshot(geodataframe_snapshot, region_coords):
    """Snapshot test for sample_risk_to_buildings function.

    This tests the complete workflow of:
    1. Calculating wind-adjusted risk for a small region
    2. Sampling risk values at building locations
    3. Ensuring the output GeoDataFrame structure is correct
    """
    # Use a small region to keep test fast
    x_slice, y_slice = region_coords

    # Calculate risk for the region
    risk_ds = calculate_wind_adjusted_risk(x_slice=x_slice, y_slice=y_slice)

    # Sample risk to buildings
    buildings_gdf = sample_risk_to_buildings(ds=risk_ds)

    assert buildings_gdf.crs is not None and buildings_gdf.crs.to_epsg() == 4326
    assert buildings_gdf.geometry.notna().all()

    # Snapshot the result
    assert geodataframe_snapshot == buildings_gdf
