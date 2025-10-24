#!/bin/bash
# COILED n-tasks 1
# COILED --region us-west-2
# COILED --vm-type m8g.large
# COILED --forward-aws-credentials
# COILED --tag project=OCR

# Download the zip
curl -L https://usfs-public.box.com/shared/static/h55qel755s97nagdu97ebd4z6fzpp3w1.zip -o RDS-2025-0006.zip

# Unzip
mkdir -p USFS_fire_risk
unzip RDS-2025-0006.zip -d USFS_fire_risk/

# Upload the tiffs to s3

# local option
# s5cmd cp --show-progress 'USFS_fire_risk/*/*/*.tif' 's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2025-0006/input_tif/'

# option for coiled
aws s3 cp USFS_fire_risk/ s3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2025-0006/input_tif/ --recursive --exclude "*" --include "*.tif"
