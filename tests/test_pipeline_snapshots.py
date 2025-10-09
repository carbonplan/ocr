import pytest

from ocr.pipeline.process_region import sample_risk_to_buildings
from ocr.risks.fire import calculate_wind_adjusted_risk

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    'region_coords',
    [
        pytest.param((slice(-120.0, -119.995), slice(35.005, 35.0)), id='california_coast'),
        pytest.param((slice(-105.0, -104.995), slice(40.005, 40.0)), id='colorado_rockies'),
        pytest.param((slice(-122.5, -122.495), slice(47.605, 47.6)), id='seattle_area'),
        pytest.param((slice(-84.4, -84.395), slice(33.755, 33.75)), id='georgia_piedmont'),
        pytest.param((slice(-111.9, -111.895), slice(33.455, 33.45)), id='arizona_desert'),
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
