# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m7a.xlarge
# COILED --tag project=OCR
# COILED --name Aggregate_Geoparquet


# aggregate geoparquet regions, reproject and write
import click
import duckdb

from ocr.template import VectorConfig
from ocr.utils import apply_s3_creds, install_load_extensions

install_load_extensions()
apply_s3_creds()


def aggregated_gpq(branch: str):
    vector_config = VectorConfig(branch=branch)

    duckdb.sql(f"""
        SET preserve_insertion_order=false;
        COPY (
        SELECT *
        FROM '{vector_config.region_geoparquet_prefix}/*.parquet')
        TO  '{vector_config.consolidated_geoparquet_uri}' (
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);""")


@click.command()
@click.option('-b', '--branch', help='data branch: [QA, prod]. Default QA')
def main(branch: str):
    aggregated_gpq(branch)


if __name__ == '__main__':
    main()
