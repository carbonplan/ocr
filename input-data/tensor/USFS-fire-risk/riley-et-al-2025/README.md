## Scripts to generate Icechunk store(s) for the USFS RDS-2025-0006 dataset.

https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0006

Spatial datasets of probabilistic wildfire risk components for the conterminous United States (270m) for circa 2011 climate and projected future climate circa 2047

### Order of operations

1. Run `bash 01_bash transfer_src_zip_to_s3.sh` to download zip archive from USFS -> unzip -> upload to s3 bucket.
2. Run `02_tiff_to_icechunk.py` to convert collection of tiffs into icechunk store(s).
