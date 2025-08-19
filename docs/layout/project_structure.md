# Project layout

This file documents the repository layout and where common code, data ingestion scripts, and outputs live. Use this as a technical reference when adding new modules or data.

Root layout (high level):

```text
ocr/                 -> Python package containing core pipeline and CLI
├── bucket_creation/  -> helper scripts to create S3 buckets and storage
│   └── create_s3_bucket.py
├── input-data/       -> ingestion scripts and raw data pointers
│   ├── tensor/       -> raster/tensor ingestion (Zarr/Icechunk)
│   │   └── USFS_fire_risk/
│   └── vector/       -> vector ingestion and subsetting
│       ├── alexandre-2016/
│       ├── calfire_structures_destroyed/
│       └── overture_vector/
├── notebooks/        -> exploratory analysis and examples (tutorials)
├── docs/             -> documentation (how-to, tutorials, references, explanations)
├── tests/            -> unit and integration tests
└── deploy/           -> CLI entry-points and deployment helpers
```

Key directories explained:

- `ocr/` — Python package. Main modules:

  - `ocr.deploy` — CLI commands that orchestrate pipeline jobs.
  - `ocr.pipeline` — implementation of region processing, aggregation, and tile creation.
  - `ocr.datasets` — dataset catalog accessor and utilities.

- `input-data/` — contains scripts that download and prepare source datasets. Treat this directory as the canonical provenance for input data.

- `notebooks/` — useful for Tutorials: convert stable notebooks into markdown tutorials under `docs/tutorials/` when they become reproducible.

- Adding new code or data:

1. Code: add new modules under `ocr/` and include unit tests in `tests/`.
2. Data ingestion: place ingestion scripts in `input-data/` and register dataset metadata in `ocr` catalog if needed.
3. Docs: update `docs/` with a short How-to describing how to run the new code and a Technical reference for public APIs.
