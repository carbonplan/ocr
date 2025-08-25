import duckdb
import geopandas as gpd
import pytest

from ocr.pipeline.aggregate import aggregated_gpq


@pytest.mark.integration
def test_aggregated_gpq_integration(region_risk_parquet):
    # Reuse config and already-generated risk region parquet
    cfg = region_risk_parquet['config']
    assert cfg.vector is not None
    vector_cfg = cfg.vector

    total_rows = 0
    for f in vector_cfg.region_geoparquet_uri.glob('*.parquet'):
        # Use DuckDB COUNT(*) defensively (some files may have many columns)
        total_rows += duckdb.sql(f"SELECT count(*) FROM '{f}'").fetchone()[0]

    aggregated_gpq(cfg)

    out_file = vector_cfg.building_geoparquet_uri
    assert out_file.exists(), 'Output parquet file was not created'
    gdf = gpd.read_parquet(out_file)

    assert len(gdf) == total_rows, f'Expected {total_rows} rows, found {len(gdf)}'
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326
    assert gdf.geometry.notna().all(), 'Geometries should all be present'

    # Idempotency
    aggregated_gpq(cfg)
    final_count2 = duckdb.sql(f"SELECT count(*) FROM '{out_file}'").fetchone()[0]
    assert final_count2 == total_rows
