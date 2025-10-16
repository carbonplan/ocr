from ocr.config import OCRConfig


def aggregated_gpq(config: OCRConfig):
    import duckdb

    from ocr.console import console
    from ocr.utils import apply_s3_creds, install_load_extensions

    connection = duckdb.connect(database=':memory:')

    input_path = config.vector.region_geoparquet_uri
    output_path = config.vector.building_geoparquet_uri
    path = input_path / '*.parquet'

    needs_s3 = any(str(p).startswith('s3://') for p in [input_path, output_path])

    install_load_extensions(aws=needs_s3, spatial=True, httpfs=True, con=connection)
    apply_s3_creds(region='us-west-2', con=connection)

    if config.debug:
        console.log(f'Aggregating geoparquet regions from: {path}')

    # connection.execute(f"""
    #     SET preserve_insertion_order=false;
    #     COPY (
    #     SELECT *
    #     FROM '{path}')
    #     TO  '{output_path}' (
    #     FORMAT 'parquet',
    #     COMPRESSION 'zstd',
    #     OVERWRITE_OR_IGNORE true);""")
    connection.execute(f"""
        SET preserve_insertion_order=false;
        COPY (
        SELECT *
        FROM (
            SELECT
                *,
                SUBSTRING(GEOID, 1, 2) AS state_fips,
                SUBSTRING(GEOID, 3, 3) AS county_fips
            FROM '{path}'
        )
        )
        TO '{output_path}' (
            FORMAT 'parquet',
            PARTITION_BY (state_fips, county_fips),
            COMPRESSION 'zstd',
            OVERWRITE_OR_IGNORE true
        );""")

    if config.debug:
        console.log(f'Aggregation complete. Consolidated geoparquet written to: {output_path}')
