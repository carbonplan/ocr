# aggregate geoparquet regions, reproject and write


from ocr.types import Branch


def aggregated_gpq(branch: Branch):
    import duckdb

    from ocr.console import console
    from ocr.template import VectorConfig
    from ocr.utils import apply_s3_creds, install_load_extensions

    install_load_extensions()
    apply_s3_creds()
    vector_config = VectorConfig(branch=branch.value)

    console.log(f'Aggregating geoparquet regions for branch: {branch.value}')
    console.log(f'Using vector config: {vector_config}')

    duckdb.sql(f"""
        SET preserve_insertion_order=false;
        COPY (
        SELECT *
        FROM '{vector_config.region_geoparquet_uri}*.parquet')
        TO  '{vector_config.consolidated_geoparquet_uri}' (
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);""")

    console.log(
        'Aggregation complete. Consolidated geoparquet written to:',
        vector_config.consolidated_geoparquet_uri,
    )
