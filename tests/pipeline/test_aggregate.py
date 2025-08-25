import geopandas as gpd
import pytest
from shapely.geometry import Point
from upath import UPath

from ocr.config import OCRConfig
from ocr.pipeline.aggregate import aggregated_gpq


def _write_region_file(path: UPath, n: int, start: int = 0):
    """Create a small geoparquet region file with n point geometries."""
    gdf = gpd.GeoDataFrame(
        {
            'id': list(range(start, start + n)),
            'value': [f'v{i}' for i in range(start, start + n)],
            'geometry': [Point(-120 + i * 0.01, 40 + i * 0.01) for i in range(n)],
        },
        crs='EPSG:4326',
    )
    gdf.to_parquet(path)
    return n


@pytest.mark.integration
def test_aggregated_gpq_integration(tmp_path):
    """Run the real aggregation using DuckDB against local geoparquet inputs.

    This avoids any mocking: we create real parquet region shards, run aggregation, then verify the
    consolidated output contains all rows. A second invocation ensures the
    COPY with OVERWRITE_OR_IGNORE executes idempotently.
    """

    root = UPath(tmp_path)
    cfg = OCRConfig(
        storage_root=str(root),
        debug=True,
        vector=None,
        icechunk=None,
        chunking=None,
        coiled=None,
    )

    # Create multiple region parquet files
    total_rows = 0
    # Ensure vector sub-config present
    assert cfg.vector is not None
    vector_cfg = cfg.vector
    total_rows += _write_region_file(vector_cfg.region_geoparquet_uri / 'region_a.parquet', 5)
    total_rows += _write_region_file(
        vector_cfg.region_geoparquet_uri / 'region_b.parquet', 3, start=100
    )

    # Run aggregation (first run)

    aggregated_gpq(cfg)

    assert cfg.vector is not None
    out_file = cfg.vector.building_geoparquet_uri
    assert out_file.exists(), 'Output parquet file was not created'

    # Use duckdb to count rows in consolidated file
    import duckdb

    # duckdb.sql(...).fetchall() returns list of tuples.
    count = duckdb.sql(f"SELECT count(*) FROM '{out_file}'").fetchall()[0][0]
    assert count == total_rows, f'Expected {total_rows} rows, found {count}'

    # Run aggregation (second run) to ensure idempotency / overwrite works
    aggregated_gpq(cfg)
    count2 = duckdb.sql(f"SELECT count(*) FROM '{out_file}'").fetchall()[0][0]
    assert count2 == total_rows, 'Row count changed after second aggregation run'
