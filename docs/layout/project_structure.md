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
    chunking_config.py # Dataset chunking config and utilites
    config.py # Contains batch job config class
    datasets.py # Catalog of input datasets with helper utils
    main.py # Coordination click app for deploying pipeline scripts
    template.py # Icechunk template chunking and writing utils
    utils.py # Geospatial helper utils
    wind.py # Wind adjustment related functions

tests/
```
