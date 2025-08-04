# aggregate geoparquet regions, reproject and write


from upath import UPath


def aggregated_gpq(input_path: UPath, output_path: UPath):
    import duckdb

    from ocr.console import console
    from ocr.utils import apply_s3_creds, install_load_extensions

    install_load_extensions()
    apply_s3_creds()
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
