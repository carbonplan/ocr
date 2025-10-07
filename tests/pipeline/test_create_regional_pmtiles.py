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

    counties_gdf = gpd.GeoDataFrame(
        {'NAME': ['TestCounty'], 'geometry': [geom], 'GEOID': ['000000000000000']}, crs='EPSG:4326'
    )
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

    # --- Preflight attribute validation using DuckDB (mirrors create_regional_pmtiles queries) ---
    # We regenerate the JSON properties directly to ensure expected keys & sample values exist
    con_check = duckdb.connect(database=':memory:')
    con_check.execute('install spatial; load spatial;')
    tract_rows = con_check.execute(f"""
        SELECT json_object(
            '7', GEOID,
            '0', building_count,
            '1', mean_wind_risk_2011,
            '2', mean_wind_risk_2047
        ) AS props
        FROM read_parquet('{cfg.vector.tracts_summary_stats_uri}')
        LIMIT 1
    """).fetchall()
    county_rows = con_check.execute(f"""
        SELECT json_object(
            '7', GEOID,
            '0', building_count,
            '1', mean_wind_risk_2011,
            '2', mean_wind_risk_2047
        ) AS props
        FROM read_parquet('{cfg.vector.counties_summary_stats_uri}')
        LIMIT 1
    """).fetchall()
    assert tract_rows and county_rows, 'No rows returned from summary stats parquet files'
    tract_row = tract_rows[0][0]
    county_row = county_rows[0][0]
    import json as _json

    tract_props = _json.loads(tract_row)
    county_props = _json.loads(county_row)
    for k in [
        '7',
        '0',
        '1',
        '2',
    ]:
        assert k in tract_props, f'Missing key {k} in tract properties JSON'
    for k in [
        '7',
        '0',
        '1',
        '2',
    ]:
        assert k in county_props, f'Missing key {k} in county properties JSON'
    assert tract_props['tract_geoid'] == '000000000000000'
    assert county_props['county_name'] == 'TestCounty'

    # Idempotency
    create_regional_pmtiles(cfg)
    assert tract_pmtiles.stat().st_size > 0
    assert county_pmtiles.stat().st_size > 0
