import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

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

    consolidated_buildings_parquet = (
        f'{config.vector.building_geoparquet_uri.parent / "consolidated-buildings.parquet"}'
    )

    if config.debug:
        console.log(f'Creating a consolidated parquet file at: {consolidated_buildings_parquet}')

    connection.execute(f"""
        SET preserve_insertion_order=false;
        COPY (SELECT * FROM '{path}')
        TO '{consolidated_buildings_parquet}'
        (FORMAT 'parquet', COMPRESSION 'zstd', OVERWRITE_OR_IGNORE true);
    """)

    connection.execute(f"""
        SET preserve_insertion_order=false;
        COPY (
            SELECT *,
                SUBSTRING(GEOID, 1, 2) AS state_fips,
                SUBSTRING(GEOID, 3, 3) AS county_fips
            FROM '{path}'
        )
        TO '{output_path}' (
            FORMAT 'parquet',
            PARTITION_BY (state_fips, county_fips),
            COMPRESSION 'zstd',
            OVERWRITE_OR_IGNORE true
        );""")

    if config.debug:
        console.log(f'partitioned buildings written to: {output_path}')

    metadata_dict = config.vector.metadata_dict

    if config.debug:
        console.log('writing _common_metadata sidecar')

    dataset = ds.dataset(str(output_path), format='parquet', partitioning='hive')

    # ignore the state / county names in metadata since they are just for the hive partition
    schema_without_partitions = dataset.schema
    partition_names = {'state_fips', 'county_fips'}
    field_indices = [
        i for i, field in enumerate(schema_without_partitions) if field.name not in partition_names
    ]
    schema_without_partitions = pa.schema(
        [schema_without_partitions.field(i) for i in field_indices]
    )

    arrow_meta = {k.encode(): v.encode() for k, v in metadata_dict.items()}
    existing_meta = schema_without_partitions.metadata or {}

    new_schema = schema_without_partitions.with_metadata({**existing_meta, **arrow_meta})

    pq.write_metadata(new_schema, f'{output_path}/_common_metadata')
    connection.close()
