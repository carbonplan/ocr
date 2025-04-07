#!/bin/bash
# COILED n-tasks 1
# COILED --region us-west-2
# COILED --vm-type m8g.large
# COILED --forward-aws-credentials
# COILED --tag project=OCR
# COILED --disk-size 80


# Download the zip
curl -L https://usfs-public.box.com/shared/static/7itw7p56vje2m0u3kqh91lt6kqq1i9l1.zip -o RDS-2020-0016-02.zip

# Unzip
unzip RDS-2020-0016-02.zip -d RDS-2020-0016-02/

# Upload the single to s3
aws s3 cp RDS-2020-0016-02/BP_CONUS/BP_CONUS.tif s3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/BP_CONUS.tif
