import pytest

from ocr.pipeline.process_region import sample_risk_to_buildings

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    'region_id',
    [
        pytest.param('california-coast', id='california-coast'),
        pytest.param('colorado-rockies', id='colorado-rockies'),
        pytest.param('seattle-area', id='seattle-area'),
        pytest.param('georgia-piedmont', id='georgia-piedmont'),
        pytest.param('arizona-desert', id='arizona-desert'),
    ],
)
def test_sample_risk_to_buildings_snapshot(
    geodataframe_snapshot, get_wind_adjusted_risk, region_id
):
    """Snapshot test for sample_risk_to_buildings function.

    This tests the complete workflow of:
    1. Calculating wind-adjusted risk for a small region (cached across tests)
    2. Sampling risk values at building locations
    3. Ensuring the output GeoDataFrame structure is correct
    """
    # Get cached risk calculation (shared with fire snapshot tests)
    risk_ds = get_wind_adjusted_risk(region_id)

    # Sample risk to buildings
    buildings_gdf = sample_risk_to_buildings(ds=risk_ds)

    assert buildings_gdf.crs is not None and buildings_gdf.crs.to_epsg() == 4326
    assert buildings_gdf.geometry.notna().all()

    # Snapshot the result
    assert geodataframe_snapshot == buildings_gdf
