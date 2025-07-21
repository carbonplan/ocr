#!/bin/bash

# COILED container quay.io/carbonplan/ocr:latest
# COILED n-tasks 1
# COILED region us-west-2
# COILED --forward-aws-credentials
# COILED --tag Project=OCR
# COILED --vm-type c7a.2xlarge
# COILED --disk-size 150


# schlep to local disk
s5cmd cp --sp 's3://carbonplan-ocr/intermediate/fire-risk/vector/CA_12_risk_scores.parquet' 'CA_12_risk_scores.parquet'

# convert to FGB
ogr2ogr -progress -f FlatGeobuf \
CA_12_risk_scores.fgb  \
CA_12_risk_scores.parquet \
-nlt PROMOTE_TO_MULTI \

echo gdal conversion done

# gen pmtiles with tons of mystery knobs
tippecanoe -o CA_12_risk_scores.pmtiles -f -P --drop-smallest-as-needed -q --extend-zooms-if-still-dropping -zg CA_12_risk_scores.fgb


echo tippecanoe tiles done

# schlep it back to s3
s5cmd cp --sp 'CA_12_risk_scores.pmtiles' 's3://carbonplan-ocr/intermediate/fire-risk/vector/CA_12_risk_scores_auto_drop_smallest.pmtiles'
