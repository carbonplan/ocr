from ocr.config import OCRConfig


def partition_buildings_by_geography(config: OCRConfig):
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
        console.log(f'Loading buildings data from: {path}')

    connection.execute(f"""
        SET preserve_insertion_order=false;
        CREATE TEMP TABLE buildings_temp AS
        SELECT
            *,
            SUBSTRING(GEOID, 1, 2) AS state_fips,
            SUBSTRING(GEOID, 3, 3) AS county_fips
        FROM '{path}';
    """)

    if config.debug:
        console.log(f'Partitioning geoparquet regions to: {output_path}')

    connection.execute(f"""
        COPY buildings_temp
        TO '{output_path}' (
            FORMAT 'parquet',
            PARTITION_BY (state_fips, county_fips),
            COMPRESSION 'zstd',
            OVERWRITE_OR_IGNORE true
        );""")

    if config.debug:
        console.log(f'Partitioned buildings written to: {output_path}')

    consolidated_buildings_parquet = (
        f'{config.vector.building_geoparquet_uri.parent / "consolidated-buildings.parquet"}'
    )

    if config.debug:
        console.log(f'Creating a consolidated parquet file at: {consolidated_buildings_parquet}')

    connection.execute(f"""
        COPY (
            SELECT * EXCLUDE (state_fips, county_fips)
            FROM buildings_temp
        )
        TO '{consolidated_buildings_parquet}'
        (
            FORMAT 'parquet',
            COMPRESSION 'zstd',
            OVERWRITE_OR_IGNORE true,
            ROW_GROUP_SIZE 10000000
        );""")

    if config.debug:
        console.log(f'Consolidated buildings written to: {consolidated_buildings_parquet}')

    connection.execute('DROP TABLE buildings_temp')
