# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m7a.xlarge
# COILED --tag project=OCR
# COILED --name Aggregate_Geoparquet


# aggregate geoparquet regions, reproject and write

import duckdb

from ocr.utils import apply_s3_creds, install_load_extensions

install_load_extensions()
apply_s3_creds()


risk = duckdb.sql("""
    SET preserve_insertion_order=false;
    COPY (
    SELECT *
    FROM 's3://carbonplan-ocr/intermediate/fire-risk/vector/PIPELINE/*.parquet')
    TO  's3://carbonplan-ocr/intermediate/fire-risk/vector/AGGREGATED_PARQUET_OUTPUT/aggregated_wind.parquet' (
    FORMAT 'parquet',
    COMPRESSION 'zstd',
    OVERWRITE_OR_IGNORE true);""")
