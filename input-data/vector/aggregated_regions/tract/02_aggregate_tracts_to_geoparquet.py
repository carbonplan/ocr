# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m8g.large
# COILED --tag project=OCR

import duckdb

from ocr.utils import apply_s3_creds, install_load_extensions

install_load_extensions()
apply_s3_creds()

duckdb.query("""COPY (SELECT * FROM read_parquet('s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/tracts/FIPS/*.parquet'))
        TO  's3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/tracts/tracts.parquet' (
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true)""")
