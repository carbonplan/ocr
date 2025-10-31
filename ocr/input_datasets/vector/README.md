# Vector Dataset Processing

This directory contains processors for vector (Parquet/GeoParquet) datasets.

## Available Datasets

### overture-maps

Overture Maps building and address data for CONUS.

**Source**: <https://overturemaps.org>
**Release**: 2025-09-24.0
**Type**: Vector (Parquet)

**Features**:

-   Buildings data with bbox and geometry
-   Addresses data with full attribute set
-   CONUS-bounded subset from global dataset

**Usage**:

```bash
# Process both buildings and addresses (default)
ocr ingest-data run-all overture-maps

# Process only buildings
ocr ingest-data process overture-maps --overture-data-type buildings

# Process only addresses
ocr ingest-data process overture-maps --overture-data-type addresses

# Dry run to preview operations
ocr ingest-data run-all overture-maps --dry-run
```

**Output**:

-   Buildings: `s3://carbonplan-ocr/input/fire-risk/vector/CONUS_overture_buildings_2025-09-24.0.parquet`
-   Addresses: `s3://carbonplan-ocr/input/fire-risk/vector/CONUS_overture_addresses_2025-09-24.0.parquet`

**Processing time**:

-   Buildings: ~14 minutes on c8g.2xlarge
-   Addresses: ~4 minutes on c8g.xlarge

## Implementation Notes

Vector processors differ from tensor processors in that they:

1. Often query directly from source S3 (no download step)
2. Use DuckDB for spatial subsetting and format conversion
3. Write directly to output S3 bucket

The Overture processor demonstrates this pattern by:

1. Skipping the download step (data is queried from Overture's S3)
2. Using DuckDB SQL to filter by CONUS bounding box
3. Writing results directly to carbonplan-ocr S3 bucket

## Adding a New Vector Dataset

See the main [Input Datasets README](../README.md) for the general template. Vector-specific considerations:

```python
class MyVectorProcessor(BaseDatasetProcessor):
    dataset_name = 'my-vector-dataset'
    dataset_type = 'vector'

    def download(self) -> None:
        # If data is already on S3, you can skip download
        console.log('Skipping download - data queried directly from S3')

    def process(self) -> None:
        # Use DuckDB, GeoPandas, or other tools
        # to process and write to S3
        pass
```
