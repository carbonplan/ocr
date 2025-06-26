## Project layout

```
.binder/
    Dockerfile  # Dockerfile containing build for s5cmd, pmtiles, gpq and tippecanoe.

bucket_creation/
    create_s3_bucket.py # Script to initialize project S3 bucket.

input-data/ # Scripts to ingest input datasets into Icechunk stores.

notebooks/ # Exploratory Jupyter Notebooks

ocr/
    pipeline/ # Data production pipeline scripts
    # ocr namespace utilities

tests/
```
