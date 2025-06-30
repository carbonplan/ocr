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

```
└── 📁ocr
    └── 📁.binder
        ├── Dockerfile # Dockerfile containing build for s5cmd, pmtiles, gpq and tippecanoe.
        └── ...
    └── 📁bucket_creation
        ├── create_s3_bucket.py # Script to initialize project S3 bucket.
    └── 📁input-data # Scripts to ingest input datasets into Icechunk stores.
        └── 📁tensor
            └── 📁USFS_fire_risk
        └── 📁vector
            └── 📁alexandre-2016
            └── 📁calfire_stuctures_destroyed
            └── 📁overture_vector
    └── 📁notebooks # Exploratory Jupyter Notebooks
    └── 📁ocr
        └── 📁pipeline # Data production pipeline scripts
            # ocr namespace utilities
            ├── __init__.py
            ├── 01_Write_Region.py
            ├── 02_Aggregate.py
            ├── 02_Pyramid.py
            ├── 03_Tiles.sh
            ├── README.md
    └── 📁tests
```
