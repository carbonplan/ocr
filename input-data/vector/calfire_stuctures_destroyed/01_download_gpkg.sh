#!/bin/bash
# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type c8g.large
# COILED --tag project=OCR


# Direct download link pulled from here: https://gis.data.ca.gov/datasets/994d3dc4569640caadbbc3198d5a3da1_0/explore?location=37.079173%2C-120.109296%2C6.33
# Download with wget, but stream stdout directly into aws cli to upload!

wget -qO- 'https://stg-arcgisazurecdataprod1.az.arcgis.com/exportfiles-9659-15353/POSTFIRE_MASTER_DATA_SHARE_-7513582634452909149.gpkg?sv=2018-03-28&sr=b&sig=RHNG4HG4ADNqRAMnP9aL5WXVBR4wAWamOBsiAC3k89Q%3D&se=2025-03-28T17%3A46%3A41Z&sp=r' | aws s3 cp - s3://carbonplan-ocr/input/fire-risk/vector/cal-fire-structures-destroyed/cal-fire-structures-destroyed.gpkg
