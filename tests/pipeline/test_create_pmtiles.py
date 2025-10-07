import shutil

import duckdb
import geopandas as gpd  # type: ignore
import pytest

from ocr.pipeline.aggregate import aggregated_gpq
from ocr.pipeline.create_pmtiles import create_pmtiles


@pytest.mark.integration
def test_create_pmtiles_end_to_end(region_risk_parquet):
    """End-to-end test for building-level PMTiles creation.

    Steps:
    1. Use region fixture to ensure a region geoparquet exists.
    2. Aggregate region geoparquets into a consolidated buildings parquet.
    3. Invoke create_pmtiles (duckdb CLI + tippecanoe pipeline) to build PMTiles.
    4. Validate that PMTiles file exists, is non-empty, and source parquet has expected columns.
    5. Re-run for idempotency.

    Skips gracefully if external binaries (tippecanoe / duckdb CLI) are not available.
    """

    if shutil.which('tippecanoe') is None or shutil.which('duckdb') is None:
        pytest.skip('tippecanoe and/or duckdb CLI not available in PATH')

    cfg = region_risk_parquet['config']

    # Aggregate region geoparquets to consolidated buildings parquet
    aggregated_gpq(cfg)
    buildings_parquet = cfg.vector.building_geoparquet_uri
    assert buildings_parquet.exists(), 'Consolidated buildings parquet missing'

    bdf = gpd.read_parquet(buildings_parquet)
    assert not bdf.empty, 'Consolidated buildings parquet is empty'
    assert set(bdf.columns) == set(
        [
            'wind_risk_2011',
            'wind_risk_2047',
            'burn_probability_2011',
            'burn_probability_2047',
            'conditional_risk_usfs',
            'burn_probability_usfs_2011',
            'burn_probability_usfs_2047',
            'geometry',
            'bbox',
        ]
    )

    # Run pipeline to create PMTiles
    create_pmtiles(cfg)
    pmtiles_path = cfg.vector.buildings_pmtiles_uri
    assert pmtiles_path.exists(), 'Buildings PMTiles not created'
    assert pmtiles_path.stat().st_size > 0, 'Buildings PMTiles file is empty'

    # Validate (indirectly) the feature properties by regenerating one JSON feature row via duckdb
    con = duckdb.connect(database=':memory:')
    con.execute('install spatial; load spatial;')
    sample_rows = con.execute(
        f"""
        SELECT json_object(
            '0', wind_risk_2011,
            '1', wind_risk_2047
        ) AS props
        FROM read_parquet('{buildings_parquet}')
        LIMIT 1
        """
    ).fetchall()
    assert sample_rows, 'No rows returned from buildings parquet during JSON regeneration'
    import json as _json

    props = _json.loads(sample_rows[0][0])
    for k in ['0', '1']:
        assert k in props, f'Missing key {k} in regenerated properties JSON'
        # Values can legitimately be 0.0; just assert they are numeric (duckdb -> python type)
        assert props[k] is None or isinstance(props[k], int | float), f'Non-numeric value for {k}'

    # Idempotency: re-run and ensure file still exists and non-zero size
    before_size = pmtiles_path.stat().st_size
    create_pmtiles(cfg)
    after_size = pmtiles_path.stat().st_size
    assert after_size > 0, 'PMTiles missing after second run'
    # Tippecanoe output can vary slightly run-to-run (timestamps, ordering). Allow small delta.
    size_delta = abs(after_size - before_size)
    assert size_delta / before_size < 0.01, (
        f'PMTiles size changed more than 1% between runs (before={before_size}, after={after_size})'
    )
