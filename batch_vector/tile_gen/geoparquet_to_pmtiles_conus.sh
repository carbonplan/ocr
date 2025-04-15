#!/bin/bash

# COILED container quay.io/carbonplan/ocr:latest
# COILED n-tasks 1
# COILED region us-west-2
# COILED --forward-aws-credentials
# COILED --tag project=OCR
# COILED --vm-type c7a.8xlarge
# COILED --disk-size 300


# schlep to local disk
s5cmd cp --sp 's3://carbonplan-ocr/intermediate/fire-risk/vector/CONUS_12_risk_scores.parquet' 'CONUS_12_risk_scores.parquet'

# convert to FGB
ogr2ogr -progress -f FlatGeobuf \
CONUS_12_risk_scores.fgb  \
CONUS_12_risk_scores.parquet \
-nlt PROMOTE_TO_MULTI \

echo gdal conversion done

# gen pmtiles with tons of mystery knobs
tippecanoe -o CONUS_12_risk_scores.pmtiles -f -P --drop-densest-as-needed -q --extend-zooms-if-still-dropping -zg CONUS_12_risk_scores.fgb


echo tippecanoe tiles done

# schlep it back to s3
s5cmd cp --sp 'CONUS_12_risk_scores.pmtiles' 's3://carbonplan-ocr/intermediate/fire-risk/vector/CONUS_12_risk_scores.pmtiles'

echo tiles moved to s3 done
