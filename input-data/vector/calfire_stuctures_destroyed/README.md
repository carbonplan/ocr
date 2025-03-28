## Scripts to generate geoparquet for the CalFire Structures Destroyed dataset

https://gis.data.ca.gov/datasets/994d3dc4569640caadbbc3198d5a3da1_0/explore?location=37.079173%2C-120.109296%2C6.33

### Order of operations

1. Run `bash 01_download_gpkg.py.sh` or `coiled batch run 01_download_gpkg.py.sh` to download the geopackage with wget and stream it to s3.
2. Run `02_gpkg_to_geoparquet.py.py` locally or with coiled to convert the gpkg to geoparquet.
3. TODO: Add geoparquet to catalog.
