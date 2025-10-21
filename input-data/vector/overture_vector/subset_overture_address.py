# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type c8g.xlarge
# COILED --tag project=OCR

import duckdb

from ocr.utils import apply_s3_creds, install_load_extensions

install_load_extensions()
apply_s3_creds()

release = '2025-09-24.0'

# conus bbox
bbox = (-125.354004, 24.413323, -66.555176, 49.196737)

result = duckdb.sql(f"""COPY (SELECT * FROM read_parquet('s3://overturemaps-us-west-2/release/{release}/theme=addresses/type=address/*.parquet')
WHERE
bbox.xmin BETWEEN {bbox[0]} AND {bbox[2]} AND
bbox.ymin BETWEEN {bbox[1]} AND {bbox[3]} ) TO 's3://carbonplan-ocr/input/fire-risk/vector/CONUS_overture_addresses_{release}.parquet'  (FORMAT 'parquet', COMPRESSION 'zstd');""")
