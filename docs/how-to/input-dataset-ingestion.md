# Input Dataset Ingestion

This guide covers how to ingest and process input datasets for the OCR (Open Climate Risk) project using the unified CLI infrastructure.

## Overview

The input dataset infrastructure provides a consistent interface for ingesting both tensor (raster/Icechunk) and vector (GeoParquet) datasets:

## Quick Start

### Discovery

List all available datasets:

```bash
pixi run ocr ingest-data list-datasets
```

### Processing

Process a dataset (always dry run first to preview):

```bash
# Preview operations (recommended first step)
pixi run ocr ingest-data run-all scott-et-al-2024 --dry-run

# Execute the full pipeline
pixi run ocr ingest-data run-all scott-et-al-2024

# Use Coiled for distributed processing
pixi run ocr ingest-data run-all scott-et-al-2024 --use-coiled
```

### Dataset-Specific Options

Different datasets support different processing options:

```bash
# Vector datasets: Overture Maps - select data type
pixi run ocr ingest-data process overture-maps --overture-data-type buildings

# Vector datasets: Census TIGER - select geography and states
pixi run ocr ingest-data process census-tiger \
  --census-geography-type tracts \
  --census-subset-states California --census-subset-states Oregon
```

## Available Datasets

### Tensor Datasets (Raster/Icechunk)

#### scott-et-al-2024

**USFS Wildfire Risk to Communities (2nd Edition)**

-   **RDS ID**: RDS-2020-0016-02
-   **Version**: 2024-V2
-   **Source**: [USFS Research Data Archive](https://www.fs.usda.gov/rds/archive/catalog/RDS-2020-0016-2)
-   **Resolution**: 30m (EPSG:4326), native 270m (EPSG:5070)
-   **Coverage**: CONUS
-   **Variables**: BP (Burn Probability), CRPS (Conditional Risk to Potential Structures), CFL (Conditional Flame Length), Exposure, FLEP4, FLEP8, RPS (Relative Proportion Spread), WHP (Wildfire Hazard Potential)

**Pipeline**:

1. Download 8 TIFF files from USFS Box (one per variable)
2. Merge TIFFs into Icechunk store (EPSG:5070, native resolution)
3. Reproject to EPSG:4326 at 30m resolution

**Usage**:

```bash
# Full pipeline
pixi run ocr ingest-data run-all scott-et-al-2024 --dry-run
pixi run ocr ingest-data run-all scott-et-al-2024 --use-coiled

# Individual steps
pixi run ocr ingest-data download scott-et-al-2024
pixi run ocr ingest-data process scott-et-al-2024 --use-coiled
```

**Outputs**:

-   Raw TIFFs: `s3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/`
-   Native Icechunk: `s3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02_all_vars_merge_icechunk/`
-   Reprojected: `s3://carbonplan-ocr/input/fire-risk/tensor/USFS/scott-et-al-2024-30m-4326.icechunk/`

---

#### riley-et-al-2025

**USFS Probabilistic Wildfire Risk - 2011 & 2047 Climate Runs**

-   **RDS ID**: RDS-2025-0006
-   **Version**: 2025
-   **Source**: [USFS Research Data Archive](https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0006)
-   **Resolution**: 30m (EPSG:4326), native 270m (EPSG:5070)
-   **Coverage**: CONUS
-   **Variables**: Multiple climate scenarios (2011 baseline, 2047 projections)

**Pipeline**:

1. Download TIFF files for both time periods
2. Process and merge into Icechunk stores
3. Reproject to EPSG:4326 at 30m resolution

**Usage**:

```bash
pixi run ocr ingest-data run-all riley-et-al-2025 --use-coiled
```

**Outputs**:

-   Reprojected: `s3://carbonplan-ocr/input/fire-risk/tensor/USFS/riley-et-al-2025-30m-4326.icechunk/`

---

#### dillon-et-al-2023

**USFS Spatial Datasets of Probabilistic Wildfire Risk Components (270m, 3rd Edition)**

-   **RDS ID**: RDS-2016-0034-3
-   **Version**: 2023
-   **Source**: [USFS Research Data Archive](https://www.fs.usda.gov/rds/archive/catalog/RDS-2016-0034-3)
-   **Resolution**: 30m (EPSG:4326), native 270m (EPSG:5070)
-   **Coverage**: CONUS
-   **Variables**: BP, FLP1-6 (Flame Length Probability levels)

**Pipeline**:

1. Download ZIP archive and extract TIFFs
2. Upload TIFFs to S3 and merge into Icechunk
3. Reproject to EPSG:4326 at 30m resolution

**Usage**:

```bash
pixi run ocr ingest-data run-all dillon-et-al-2023 --use-coiled
```

**Outputs**:

-   Raw TIFFs: `s3://carbonplan-ocr/input/fire-risk/tensor/USFS/dillon-et-al-2023/raw-input-tiffs/`
-   Native Icechunk: `s3://carbonplan-ocr/input/fire-risk/tensor/USFS/dillon-et-al-2023/processed-270m-5070.icechunk/`
-   Reprojected: `s3://carbonplan-ocr/input/fire-risk/tensor/USFS/dillon-et-al-2023/processed-30m-4326.icechunk/`

---

### Vector Datasets (GeoParquet)

#### overture-maps

**Overture Maps Building and Address Data for CONUS**

-   **Release**: 2025-09-24.0
-   **Source**: [Overture Maps Foundation](https://overturemaps.org)
-   **Format**: GeoParquet (WKB geometry, zstd compression)
-   **Coverage**: CONUS (spatially filtered from global dataset)
-   **Data Types**: Buildings (bbox + geometry), Addresses (full attributes), Region-Tagged Buildings (buildings + census identifiers)

**Pipeline**:

1. Query Overture S3 bucket directly (no download step)
2. Filter by CONUS bounding box using DuckDB
3. Write subsetted data to carbonplan-ocr S3 bucket
4. (If buildings processed) Perform spatial join with US Census blocks to add geographic identifiers

**Region-Tagged Buildings Processing**:

When buildings are processed, an additional dataset is automatically created that tags each building with census geographic identifiers:

-   Loads census FIPS lookup table for state/county names
-   Creates spatial indexes on buildings and census blocks
-   Performs bbox-filtered spatial join using `ST_Intersects`
-   Adds identifiers at multiple administrative levels: state, county, tract, block group, and block

**Usage**:

```bash
# Both buildings and addresses (default)
# Also creates region-tagged buildings automatically
pixi run ocr ingest-data run-all overture-maps

# Only buildings (also creates region-tagged buildings)
pixi run ocr ingest-data process overture-maps --overture-data-type buildings

# Only addresses (no region tagging)
pixi run ocr ingest-data process overture-maps --overture-data-type addresses

# Dry run
pixi run ocr ingest-data run-all overture-maps --dry-run

# Use Coiled for distributed processing
pixi run ocr ingest-data run-all overture-maps --use-coiled
```

**Outputs**:

-   Buildings: `s3://carbonplan-ocr/input/fire-risk/vector/overture-maps/CONUS-overture-buildings-2025-09-24.0.parquet`
-   Addresses: `s3://carbonplan-ocr/input/fire-risk/vector/overture-maps/CONUS-overture-addresses-2025-09-24.0.parquet`
-   Region-Tagged Buildings: `s3://carbonplan-ocr/input/fire-risk/vector/overture-maps/CONUS-overture-region-tagged-buildings-2025-09-24.0.parquet`

**Region-Tagged Buildings Schema**:

-   `geometry`: Building geometry (WKB)
-   `bbox`: Building bounding box
-   `block_geoid`: Full 15-digit census block GEOID
-   `block_group_geoid`: 12-digit block group GEOID
-   `tract_geoid`: 11-digit tract GEOID
-   `county_geoid`: 5-digit county GEOID
-   `state_fips`: 2-digit state FIPS code
-   `county_fips`: 3-digit county FIPS code
-   `tract_fips`: 6-digit tract FIPS code
-   `block_group_fips`: 1-digit block group code
-   `block_fips`: 4-digit block code
-   `state_abbrev`: State abbreviation (e.g., "CA")
-   `county_name`: County name

---

#### census-tiger

**US Census TIGER/Line Geographic Boundaries**

-   **Vintage**: 2024 (tracts/counties), 2025 (blocks)
-   **Source**: [US Census Bureau TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
-   **Format**: GeoParquet (WKB geometry, zstd compression, schema v1.1.0)
-   **Coverage**: CONUS + DC (49 states/territories, excludes Alaska & Hawaii)
-   **Geography Types**: Blocks, Tracts, Counties

**Pipeline**:

1. Download TIGER/Line shapefiles from Census Bureau (per-state for blocks/tracts)
2. Convert to GeoParquet with spatial metadata
3. Aggregate tract files using DuckDB

**Usage**:

```bash
# All geography types (default)
pixi run ocr ingest-data run-all census-tiger

# Only counties
pixi run ocr ingest-data process census-tiger --census-geography-type counties

# Tracts for specific states
pixi run ocr ingest-data process census-tiger --census-geography-type tracts \
  --census-subset-states California --census-subset-states Oregon

# Dry run
pixi run ocr ingest-data run-all census-tiger --dry-run
```

**Outputs**:

-   Blocks: `s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/blocks/blocks.parquet`
-   Tracts (per-state): `s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/tracts/FIPS/FIPS_*.parquet`
-   Tracts (aggregated): `s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/tracts/tracts.parquet`
-   Counties: `s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/counties/counties.parquet`

**Processing Time**:

-   Blocks: ~20-30 minutes (all 49 states)
-   Tracts: ~15-20 minutes (all 49 states + aggregation)
-   Counties: ~2 minutes

## CLI Reference

### Commands

-   **`list-datasets`**: Show all available datasets
-   **`download <dataset>`**: Download raw source data (tensor datasets only)
-   **`process <dataset>`**: Process and upload to S3/Icechunk
-   **`run-all <dataset>`**: Complete pipeline (download + process + cleanup)

### Global Options

-   **`--dry-run`**: Preview operations without executing (recommended before any real run)
-   **`--debug`**: Enable debug logging for troubleshooting

### Tensor Dataset Options

-   **`--use-coiled`**: Use Coiled for distributed processing (USFS datasets)

### Vector Dataset Options

#### Overture Maps

-   **`--overture-data-type <type>`**: Which data to process
    -   `buildings`: Only building geometries
    -   `addresses`: Only address points
    -   `both`: Both datasets (default)

#### Census TIGER

-   **`--census-geography-type <type>`**: Which geography to process
    -   `blocks`: Census blocks
    -   `tracts`: Census tracts (per-state + aggregated)
    -   `counties`: County boundaries
    -   `all`: All three types (default)
-   **`--census-subset-states <state> [<state> ...]`**: Process only specific states
    -   Repeat option for each state: `--census-subset-states California --census-subset-states Oregon`
    -   Use full state names (case-sensitive): `California`, `Oregon`, `Washington`, etc.

## Configuration

### Environment Variables

All settings can be overridden via environment variables:

```bash
# S3 configuration
export OCR_INPUT_DATASET_S3_BUCKET=my-bucket
export OCR_INPUT_DATASET_S3_REGION=us-east-1
export OCR_INPUT_DATASET_BASE_PREFIX=custom/prefix

# Processing options
export OCR_INPUT_DATASET_CHUNK_SIZE=16384
export OCR_INPUT_DATASET_DEBUG=true

# Temporary storage
export OCR_INPUT_DATASET_TEMP_DIR=/path/to/temp
```

### Configuration Class

The `InputDatasetConfig` class (Pydantic model) provides:

-   Type validation for all settings
-   Automatic environment variable loading (prefix: `OCR_INPUT_DATASET_`)
-   Default values for all options
-   Case-insensitive environment variable names

## Troubleshooting

### Dry Run First

Always test with `--dry-run` before executing:

```bash
ocr ingest-data run-all <dataset> --dry-run
```

This previews all operations without making changes.
