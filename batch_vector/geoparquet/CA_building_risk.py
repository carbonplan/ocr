# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type r8g.xlarge
# COILED --tag project=OCR

import duckdb

from ocr.utils import apply_s3_creds, install_load_extensions

install_load_extensions()
apply_s3_creds()

# CA subset
bbox = (-124.683838, 32.495145, -114.049072, 41.975680)

result = duckdb.sql(
    f"""SET preserve_insertion_order=false; COPY ( SELECT geometry,
    round(RANDOM(), 3) as risk_score1,
    round(RANDOM(), 3) as risk_score2,
    round(RANDOM(), 3) as risk_score3,
    round(RANDOM(), 3) as risk_score4,
    round(RANDOM(), 3) as risk_score5,
    round(RANDOM(), 3) as risk_score6,
    round(RANDOM(), 3) as risk_score7,
    round(RANDOM(), 3) as risk_score8,
    round(RANDOM(), 3) as risk_score9,
    round(RANDOM(), 3) as risk_score10,
    round(RANDOM(), 3) as risk_score11,
    round(RANDOM(), 3) as risk_score12,
        FROM 's3://carbonplan-ocr/input/fire-risk/vector/CONUS_overture_buildings_2025-03-19.1.parquet' WHERE
        bbox.xmin BETWEEN {bbox[0]} AND {bbox[2]} AND
        bbox.ymin BETWEEN {bbox[1]} AND {bbox[3]})
     TO  's3://carbonplan-ocr/intermediate/fire-risk/vector/CA_12_risk_scores.parquet' (
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);"""
)
