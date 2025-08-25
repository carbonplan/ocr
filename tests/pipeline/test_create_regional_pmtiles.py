import shutil

import duckdb
import geopandas as gpd  # type: ignore
import pytest
from shapely.ops import unary_union  # type: ignore

from ocr.pipeline.aggregate import aggregated_gpq
from ocr.pipeline.create_regional_pmtiles import create_regional_pmtiles
from ocr.pipeline.fire_wind_risk_regional_aggregator import (
    create_summary_stat_tmp_tables,
    custom_histogram_query,
)


@pytest.mark.integration
def test_create_regional_pmtiles_end_to_end(region_risk_parquet, tmp_path):
    """End-to-end test for PMTiles creation without monkeypatching.

    Uses the region_risk_parquet fixture to build a consolidated buildings parquet,
    generates minimal county/tract summary stats via DuckDB, then invokes the PMTiles
    creation pipeline. Skips if required external binaries (tippecanoe, duckdb CLI) are absent.
    """

    # Skip if external binaries not available
    if shutil.which('tippecanoe') is None or shutil.which('duckdb') is None:
        pytest.skip('tippecanoe and/or duckdb CLI not available in PATH')

    cfg = region_risk_parquet['config']

    # Ensure consolidated building parquet exists
    aggregated_gpq(cfg)
    buildings_path = cfg.vector.building_geoparquet_uri
    assert buildings_path.exists()

    b_gdf = gpd.read_parquet(buildings_path)
    assert len(b_gdf) > 0
    union = unary_union(b_gdf.geometry)
    geom = union.buffer(0.0001)

    counties_gdf = gpd.GeoDataFrame({'NAME': ['TestCounty'], 'geometry': [geom]}, crs='EPSG:4326')
    tracts_gdf = gpd.GeoDataFrame(
        {'GEOID': ['000000000000000'], 'geometry': [geom]}, crs='EPSG:4326'
    )

    counties_path = tmp_path / 'counties.parquet'
    tracts_path = tmp_path / 'tracts.parquet'
    counties_gdf.to_parquet(counties_path)
    tracts_gdf.to_parquet(tracts_path)

    con = duckdb.connect(database=':memory:')
    con.execute('install spatial; load spatial;')

    create_summary_stat_tmp_tables(
        con=con,
        counties_path=counties_path,
        tracts_path=tracts_path,
        consolidated_buildings_path=buildings_path,
    )
    hist_bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    custom_histogram_query(
        con=con,
        geo_table_name='county',
        summary_stats_path=cfg.vector.counties_summary_stats_uri,
        hist_bins=hist_bins,
    )
    custom_histogram_query(
        con=con,
        geo_table_name='tract',
        summary_stats_path=cfg.vector.tracts_summary_stats_uri,
        hist_bins=hist_bins,
    )

    # Sanity check summary stats
    assert cfg.vector.counties_summary_stats_uri.exists()
    assert cfg.vector.tracts_summary_stats_uri.exists()

    # Run PMTiles creation (invokes duckdb CLI + tippecanoe)
    create_regional_pmtiles(cfg)

    tract_pmtiles = cfg.vector.tracts_pmtiles_uri
    county_pmtiles = cfg.vector.counties_pmtiles_uri
    assert tract_pmtiles.exists(), 'Tract PMTiles not created'
    assert county_pmtiles.exists(), 'County PMTiles not created'
    assert tract_pmtiles.stat().st_size > 0, 'Tract PMTiles empty'
    assert county_pmtiles.stat().st_size > 0, 'County PMTiles empty'

    # --- Inspect PMTiles binary for expected attribute keys (lightweight validation) ---
    tract_bytes = tract_pmtiles.read_bytes()
    county_bytes = county_pmtiles.read_bytes()

    tract_expected_keys = [
        'tract_geoid',
        'building_count',
        'avg_risk_2011_horizon_1',
        'risk_2011_horizon_1',
        'wind_risk_2011_horizon_1',
    ]
    county_expected_keys = [
        'county_name',
        'building_count',
        'avg_risk_2011_horizon_1',
        'risk_2011_horizon_1',
        'wind_risk_2011_horizon_1',
    ]
    for key in tract_expected_keys:
        assert key.encode('utf-8') in tract_bytes, f'Missing key {key} in tract pmtiles'
    for key in county_expected_keys:
        assert key.encode('utf-8') in county_bytes, f'Missing key {key} in county pmtiles'

    # Presence of feature name values
    assert b'TestCounty' in county_bytes, 'County name value not found in county pmtiles'
    # Tract geoid value
    assert b'000000000000000' in tract_bytes, 'Tract GEOID value not found in tract pmtiles'

    # Idempotency
    create_regional_pmtiles(cfg)
    assert tract_pmtiles.stat().st_size > 0
    assert county_pmtiles.stat().st_size > 0
