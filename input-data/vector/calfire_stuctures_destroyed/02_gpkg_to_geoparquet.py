# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type c8g.xlarge
# COILED --tag project=OCR

import duckdb

from ocr.utils import apply_s3_creds, install_load_extensions

install_load_extensions()
apply_s3_creds()

duckdb.sql("""COPY (SELECT * FROM st_read('s3://carbonplan-ocr/input/fire-risk/vector/cal-fire-structures-destroyed/cal-fire-structures-destroyed.gpkg'))
           TO 's3://carbonplan-ocr/input/fire-risk/vector/cal-fire-structures-destroyed/cal-fire-structures-destroyed.parquet'
           (
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);""")
