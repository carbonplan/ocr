# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type r8g.xlarge
# COILED --tag Project=OCR

import duckdb

from ocr.utils import apply_s3_creds, install_load_extensions

install_load_extensions()
apply_s3_creds()

# CA subset
bbox = (-124.683838, 32.495145, -114.049072, 41.975680)

result = duckdb.sql(
    """SET preserve_insertion_order=false; COPY ( SELECT *, ST_Centroid(geometry) AS building_centroid
    FROM 's3://carbonplan-ocr/input/fire-risk/vector/CONUS_overture_buildings_2025-03-19.1.parquet'
     TO  's3://carbonplan-ocr/intermediate/fire-risk/vector/CONUS_overture_buildings_with_centroid_2025-03-19.1.parquet' (
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);"""
)
