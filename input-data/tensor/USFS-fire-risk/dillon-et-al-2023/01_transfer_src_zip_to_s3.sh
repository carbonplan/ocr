#!/bin/bash
# COILED n-tasks 1
# COILED --region us-west-2
# COILED --vm-type m8g.2xlarge
# COILED --forward-aws-credentials
# COILED --tag project=OCR

# Download the zip
curl -L <add link> -o RDS-2016-0034-3.zip

# Unzip
mkdir -p RDS-2016-0034-3
unzip RDS-2016-0034-3.zip -d RDS-2016-0034-3/

# Upload the tiffs to s3

aws s3 cp RDS-2016-0034-3/Data/I_FSim_CONUS_LF2020_270m/ s3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2016-0032-3/input_tif/ --recursive --exclude "*" --include "*.tif"
