# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m7a.xlarge
# COILED --tag project=OCR


# aggregate geoparquet regions, reproject and write

import duckdb

# Hardcoding LA region icechunk store, eventually we will use 'template' in catalog
region_id = 'y10_x2'

# we will probably want to wildcard later to aggregate
# for now, we are using a single region and just selecting out the 4326 geometry for pmtiles
risk = duckdb.sql(f"""INSTALL SPATIAL; LOAD SPATIAL; INSTALL HTTPFS; LOAD HTTPFS;
    SET preserve_insertion_order=false;
    COPY (
    SELECT
    BP as risk,
    bbox_4326 as bbox,
    geometry_4326 as geometry
    FROM 's3://carbonplan-ocr/intermediate/fire-risk/vector/{region_id}.parquet')
    TO  's3://carbonplan-ocr/intermediate/fire-risk/vector/aggregated_regions.parquet' (
    FORMAT 'parquet',
    COMPRESSION 'zstd',
    OVERWRITE_OR_IGNORE true);""")
