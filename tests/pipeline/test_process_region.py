import geopandas as gpd
import pytest


@pytest.mark.integration
def test_calculate_risk_process_region_end_to_end(region_risk_parquet):
    data = region_risk_parquet
    data['config']
    out_file = data['path']

    assert out_file.exists(), 'Expected region geoparquet output file to exist'
    gdf = gpd.read_parquet(out_file)

    expected_vars = {
        'wind_risk_2011',
        'wind_risk_2047',
        'burn_probability_2011',
        'burn_probability_2047',
        'conditional_risk_usfs',
        'burn_probability_usfs_2011',
        'burn_probability_usfs_2047',
    }
    assert expected_vars.issubset(gdf.columns), (
        f'Missing expected risk columns: {expected_vars - set(gdf.columns)}'
    )
    assert not gdf[list(expected_vars)].isna().any().any(), 'NaNs present in risk columns'
    for col in expected_vars:
        assert (gdf[col].between(0, 100)).all(), f'Values out of range in {col}'
    assert len(gdf) > 0
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326
    assert gdf.geometry.notna().all(), 'Geometries should all be present'
