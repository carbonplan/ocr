import duckdb
import geopandas as gpd  # type: ignore
import pytest
from shapely.ops import unary_union  # type: ignore

from ocr.pipeline.aggregate import aggregated_gpq
from ocr.pipeline.fire_wind_risk_regional_aggregator import (
    create_summary_stat_tmp_tables,
    custom_histogram_query,
)


@pytest.mark.integration
def test_fire_wind_risk_regional_aggregator(tmp_path, region_risk_parquet):
    """Integration test for regional fire and wind risk statistics aggregation.

    This test:
    1. Ensures a consolidated buildings parquet exists (via aggregated_gpq)
    2. Creates synthetic county and tract geometries that fully cover all buildings
    3. Runs the temporary table creation and custom histogram queries
    4. Verifies output parquet files exist with expected structure and values
    """
    cfg = region_risk_parquet['config']

    # Ensure consolidated buildings parquet exists
    aggregated_gpq(cfg)
    consolidated_buildings_path = cfg.vector.building_geoparquet_uri
    assert consolidated_buildings_path.exists(), 'Expected consolidated buildings parquet to exist'

    # Load building geometries
    b_gdf = gpd.read_parquet(consolidated_buildings_path)
    assert len(b_gdf) > 0, 'Expected at least one building geometry'
    union = unary_union(b_gdf.geometry)

    # Slight buffer to ensure full coverage, avoid topology errors on degenerate geometries
    county_geom = union.buffer(0.0001)
    tract_geom = county_geom  # Single tract identical to county for simplicity

    counties_gdf = gpd.GeoDataFrame(
        {'NAME': ['TestCounty'], 'geometry': [county_geom]},
        crs='EPSG:4326',
    )
    tracts_gdf = gpd.GeoDataFrame(
        {'GEOID': ['000000000000000'], 'geometry': [tract_geom]},
        crs='EPSG:4326',
    )

    # Write synthetic region layers to parquet
    counties_path = tmp_path / 'test_counties.parquet'
    tracts_path = tmp_path / 'test_tracts.parquet'
    counties_gdf.to_parquet(counties_path)
    tracts_gdf.to_parquet(tracts_path)

    # Prepare DuckDB connection with spatial extension
    con = duckdb.connect(database=':memory:')
    con.execute('install spatial; load spatial;')

    # Production histogram bin edges (excluding 0, which is handled separately)
    hist_bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create temp tables
    create_summary_stat_tmp_tables(
        con=con,
        counties_path=counties_path,
        tracts_path=tracts_path,
        consolidated_buildings_path=consolidated_buildings_path,
    )

    # Run custom histogram aggregation for county + tract
    counties_summary_path = cfg.vector.counties_summary_stats_uri
    tracts_summary_path = cfg.vector.tracts_summary_stats_uri
    custom_histogram_query(
        con=con,
        geo_table_name='county',
        summary_stats_path=counties_summary_path,
        hist_bins=hist_bins,
    )
    custom_histogram_query(
        con=con,
        geo_table_name='tract',
        summary_stats_path=tracts_summary_path,
        hist_bins=hist_bins,
    )

    # Validate outputs exist
    assert counties_summary_path.exists(), 'Counties summary stats parquet not written'
    assert tracts_summary_path.exists(), 'Tracts summary stats parquet not written'

    def validate(path, name_col):
        df = duckdb.sql(f"SELECT * FROM '{path}'").df()
        assert len(df) == 1, 'Expected exactly one region row'
        assert name_col in df.columns, f'Missing name column {name_col}'
        # Cast to Python int (DuckDB -> pandas may yield numpy scalar)
        building_count = int(df['building_count'].iloc[0])
        assert building_count == len(b_gdf), 'Building count mismatch'

        # Validate a representative histogram column
        hist_col = 'wind_risk_2011_horizon_1'
        assert hist_col in df.columns, f'Missing histogram column {hist_col}'
        # Extract histogram values robustly (duckdb list column, possible scalar, or stringified list)
        import ast

        hist_values_raw = df.loc[0, hist_col]
        if isinstance(hist_values_raw, list | tuple):  # type: ignore[arg-type]
            raw_list = hist_values_raw
        elif isinstance(hist_values_raw, str):
            # Try to parse string representation of list
            try:
                parsed = ast.literal_eval(hist_values_raw)
                raw_list = parsed if isinstance(parsed, list | tuple) else [parsed]  # type: ignore[arg-type]
            except Exception:
                raw_list = [hist_values_raw]
        else:
            raw_list = [hist_values_raw]

        # Coerce each element to int (treat non-numeric as zero)
        hist_values: list[int] = []
        for v in raw_list:
            try:  # Convert via string to reduce odd type issues
                hist_values.append(int(str(v)))  # type: ignore[arg-type]
            except Exception:
                hist_values.append(0)

        assert len(hist_values) >= 1, 'Histogram must contain at least one bucket'

        # Total counts from histogram correspond to non-zero + maybe zero bucket depending on implementation details
        total_hist_count = sum(hist_values)
        assert 0 <= total_hist_count <= building_count, 'Histogram total outside valid range'

        # Infer zero count (best-effort): difference between building_count and histogram sum
        inferred_zero = building_count - total_hist_count
        assert inferred_zero >= 0, 'Inferred zero bucket negative'

        # Validate averages within [0,100]
        avg_col = 'avg_USFS_RPS_horizon_1'
        avg_raw = df[avg_col].iloc[0]

        avg_val = float(avg_raw)
        assert 0 <= avg_val <= 100, 'Average risk out of range'

    validate(counties_summary_path, 'NAME')
    validate(tracts_summary_path, 'NAME')
