# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type r8g.xlarge
# COILED --tag Project=OCR

import duckdb

from ocr.utils import apply_s3_creds, install_load_extensions

install_load_extensions()
apply_s3_creds()

# CONVERT EPSG:4326 TO EPSG:5070

result = duckdb.sql(
    """SET preserve_insertion_order=false; COPY ( SELECT geometry as geometry_4326,
            bbox as bbox_4326,
            ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true) AS geometry,
            STRUCT_PACK(
                xmin := ST_XMin(ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true)),
                ymin := ST_YMin(ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true)),
                xmax := ST_XMax(ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true)),
                ymax := ST_YMax(ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true))
            ) AS bbox
    FROM 's3://carbonplan-ocr/input/fire-risk/vector/CONUS_overture_buildings_2025-03-19.1.parquet')
     TO  's3://carbonplan-ocr/input/fire-risk/vector/CONUS_overture_buildings_5070_2025-03-19.1.parquet' (
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);"""
)
