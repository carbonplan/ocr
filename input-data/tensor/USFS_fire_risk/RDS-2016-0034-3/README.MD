## Scripts to generate Icechunk store(s) for the RDS-2016-0034-3 dataset.

https://doi.org/10.2737/RDS-2016-0034-3

### Order of operations

1. Run `bash 01_bash transfer_src_zip_to_s3.sh` to download zip archive from USFS -> unzip -> upload to s3 bucket.
2. Run `02_tiff_to_icechunk.py` to convert collection of tiffs into icechunk store(s).
