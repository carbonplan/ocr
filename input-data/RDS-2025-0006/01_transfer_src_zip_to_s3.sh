#!/bin/bash

# Spatial datasets of probabilistic wildfire risk components for the conterminous United States (270m) for circa 2011 climate and projected future climate circa 2047
# This script downloads the dataset listed on this USFS page: https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0006
# The tiff files in the dataset are compressed into a zipped file. This script will download the zip archive, unzip it and upload it to s3.

# Download the zip
curl -L https://usfs-public.box.com/shared/static/h55qel755s97nagdu97ebd4z6fzpp3w1.zip -o RDS-2025-0006.zip

# Unzip
mkdir -p USFS_fire_risk
unzip RDS-2025-0006.zip -d USFS_fire_risk/

# Upload the tiffs to s3
s5cmd cp  'USFS_fire_risk/*/*/*.tif' 's3://carbonplan-ocr/input_data/tensor/USFS/RDS-2025-0006/input_tif/'
