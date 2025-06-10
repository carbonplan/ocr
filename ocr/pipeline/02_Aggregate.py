# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m7a.xlarge
# COILED --tag project=OCR


# aggregate geoparquet regions, reproject and write

import duckdb

from ocr.utils import apply_s3_creds, install_load_extensions

install_load_extensions()
apply_s3_creds()


# we will probably want to wildcard later to aggregate
# for now, we are using a single region and just selecting out the 4326 geometry for pmtiles
risk = duckdb.sql("""
    SET preserve_insertion_order=false;
    COPY (
    SELECT
    wind_risk as wind_risk,
    USFS_RPS as USFS_RPS,
    bbox_4326 as bbox,
    geometry_4326 as geometry
    FROM 's3://carbonplan-ocr/intermediate/fire-risk/vector/PIPELINE/*.parquet')
    TO  's3://carbonplan-ocr/intermediate/fire-risk/vector/AGGREGATED_PARQUET_OUTPUT/aggregated_wind.parquet' (
    FORMAT 'parquet',
    COMPRESSION 'zstd',
    OVERWRITE_OR_IGNORE true);""")
