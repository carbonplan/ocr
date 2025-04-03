## Scripts to generate Icechunk store(s) for the USFS RDS-2020-0016 dataset.

Wildfire Risk to Communities: Spatial datasets of landscape-wide wildfire risk components for the United States (2nd Edition)
https://www.fs.usda.gov/rds/archive/catalog/RDS-2020-0016

### Order of operations

1. Run `bash 01_bash transfer_src_zip_to_s3.sh` to download zip archive from USFS -> unzip -> upload to s3 bucket.
2. Run `02_tiff_to_icechunk.py` to convert collection of tiffs into icechunk store(s).
