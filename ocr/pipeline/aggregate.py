# aggregate geoparquet regions, reproject and write


from ocr.config import OCRConfig


def aggregated_gpq(config: OCRConfig):
    import duckdb

    from ocr.console import console
    from ocr.utils import apply_s3_creds, install_load_extensions

    install_load_extensions()
    apply_s3_creds()

    input_path = config.vector.region_geoparquet_uri
    output_path = config.vector.consolidated_geoparquet_uri
    path = input_path / '*.parquet'

    console.log(f'Aggregating geoparquet regions from: {path}')

    duckdb.sql(f"""
        SET preserve_insertion_order=false;
        COPY (
        SELECT *
        FROM '{path}')
        TO  '{output_path}' (
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);""")

    console.log(f'Aggregation complete. Consolidated geoparquet written to: {output_path}')
