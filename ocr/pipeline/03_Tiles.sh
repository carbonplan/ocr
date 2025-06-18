#!/bin/bash

# COILED container quay.io/carbonplan/ocr:latest
# COILED n-tasks 1
# COILED region us-west-2
# COILED --forward-aws-credentials
# COILED --tag project=OCR
# COILED --vm-type c7a.xlarge
# COILED --name Create_PMTiles


# maybe we should switch to gpq:
# ex: gpq convert --to=geojson in.geoparquet | tippecanoe -o out.pmtiles

s5cmd cp --sp 's3://carbonplan-ocr/intermediate/fire-risk/vector/AGGREGATED_PARQUET_OUTPUT/aggregated_wind.parquet' 'region.parquet'

# convert to FGB
ogr2ogr -progress -f FlatGeobuf \
region.fgb  \
region.parquet \
-nlt PROMOTE_TO_MULTI \

echo gdal conversion done

# gen pmtiles with tons of mystery knobs
tippecanoe -o aggregated.pmtiles -l risk -n "USFS BP Risk" -f -P --drop-smallest-as-needed -q --extend-zooms-if-still-dropping -zg region.fgb


echo tippecanoe tiles done

# schlep it back to s3
s5cmd cp --sp 'aggregated.pmtiles' 's3://carbonplan-ocr/intermediate/fire-risk/vector/multi_year_wind.pmtiles'
