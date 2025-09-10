# Tracts Pipeline

This directory contains a CLI utility `tracts_pipeline.py` that downloads TIGER/Line
2024 Census Tract geometries (CONUS + DC) to per–state Parquet files on S3 and then
aggregates them into a single Parquet.

## Overview

Steps performed:

1. Stream each state ZIP directly from the Census TIGER site.
2. Convert to Parquet with geometry preserved (WKB, bounding box metadata, zstd compression).
3. Aggregate all per–state Parquet files into `tracts.parquet` using DuckDB.

Default S3 layout:

```text
s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/tracts/
 FIPS/
  FIPS_06.parquet
  FIPS_41.parquet
  ...
 tracts.parquet
```

## CLI Help

Show available options:

```bash
pixi run python input-data/vector/aggregated_regions/tract/tracts_pipeline.py --help
```

List state keys (names used with --states / --exclude-states):

```bash
pixi run python input-data/vector/aggregated_regions/tract/tracts_pipeline.py --list-states
```

## Common Usage

Run full pipeline (download all states + aggregate):

```bash
pixi run python input-data/vector/aggregated_regions/tract/tracts_pipeline.py
```

Only process a subset (e.g., California + Oregon + Washington):

```bash
pixi run python input-data/vector/aggregated_regions/tract/tracts_pipeline.py \
 --states California Oregon Washington
```

Exclude a state (e.g., skip Texas):

```bash
pixi run python input-data/vector/aggregated_regions/tract/tracts_pipeline.py \
 --exclude-states Texas
```

Skip downloads (re-use existing per–state files) and just re-aggregate:

```bash
pixi run python input-data/vector/aggregated_regions/tract/tracts_pipeline.py \
 --skip-download
```

Only aggregate if the aggregate file is missing:

```bash
pixi run python input-data/vector/aggregated_regions/tract/tracts_pipeline.py \
 --aggregate-only-if-missing
```

Disable overwriting per–state files (will still error if they do not exist):

```bash
pixi run python input-data/vector/aggregated_regions/tract/tracts_pipeline.py \
 --overwrite-state False
```

Change TIGER/Line vintage year (if updating in future):

```bash
pixi run python input-data/vector/aggregated_regions/tract/tracts_pipeline.py --year 2025
```

Custom output base path (must include `FIPS/` subdir implicitly):

```bash
pixi run python input-data/vector/aggregated_regions/tract/tracts_pipeline.py \
 --output-base s3://carbonplan-ocr-dev/scratch/tracts
```

## Running on Coiled

The script is lightweight and typically runs fine locally, but you can also execute it
on a Coiled ephemeral environment for isolation or to leverage standardized configuration.

1. Submit as a Coiled job (recommended for reproducibility)

   - Create a small wrapper script or pass the module directly.
   - Ensure AWS credentials are forwarded (the script calls `apply_s3_creds`).

Example job submission (from a shell where `coiled` is configured):

```bash
pixi run coiled run \
 --name tracts-pipeline \
 --region us-west-2 \
 --tag Project=OCR \
 "python input-data/vector/aggregated_regions/tract/tracts_pipeline.py"
```

Process only the western states in a job:

```bash
pixi run coiled run \
 --name tracts-pipeline-west \
 --region us-west-2 \
 --tag Project=OCR \
 "python input-data/vector/aggregated_regions/tract/tracts_pipeline.py --states California Oregon Washington Nevada Idaho Arizona Utah Montana Wyoming Colorado NewMexico"
```

## Failure Recovery

If the run fails partway through downloads:

1. Re-run with `--skip-download` to attempt just aggregation (if enough states completed).
2. Or re-run normally; states will be overwritten unless you set `--overwrite-state False`.

If aggregation failed but per–state files exist, just re-run with `--skip-download`.
