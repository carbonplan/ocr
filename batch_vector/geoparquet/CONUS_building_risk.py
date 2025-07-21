# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type r8g.2xlarge
# COILED --tag Project=OCR

import duckdb

from ocr.utils import apply_s3_creds, install_load_extensions

install_load_extensions()
apply_s3_creds()


result = duckdb.sql(
    """SET preserve_insertion_order=false; COPY ( SELECT geometry,
    round(RANDOM(), 2) as risk_score1,
    round(RANDOM(), 2) as risk_score2,
    round(RANDOM(), 2) as risk_score3,
    round(RANDOM(), 2) as risk_score4,
    round(RANDOM(), 2) as risk_score5,
    round(RANDOM(), 2) as risk_score6,
    round(RANDOM(), 2) as risk_score7,
    round(RANDOM(), 2) as risk_score8,
    round(RANDOM(), 2) as risk_score9,
    round(RANDOM(), 2) as risk_score10,
    round(RANDOM(), 2) as risk_score11,
    round(RANDOM(), 2) as risk_score12,
        FROM 's3://carbonplan-ocr/input/fire-risk/vector/CONUS_overture_buildings_2025-03-19.1.parquet')
     TO  's3://carbonplan-ocr/intermediate/fire-risk/vector/CONUS_12_risk_scores.parquet' (
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);"""
)
