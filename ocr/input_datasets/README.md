# Input Dataset Processing

This directory contains the infrastructure for ingesting and processing input datasets for the OCR project.

## Quick Start

List available datasets:

```bash
ocr ingest-data list-datasets
```

Process a dataset (dry run first to preview):

```bash
# Preview operations
ocr ingest-data run-all scott-et-al-2024 --dry-run

# Actually run
ocr ingest-data run-all scott-et-al-2024

# Use Coiled for distributed processing
ocr ingest-data run-all scott-et-al-2024 --use-coiled
```

## Architecture

### Base Classes

-   **`BaseDatasetProcessor`**: Abstract base class for all dataset processors

    -   `download()`: Download raw source data
    -   `process()`: Process and upload to S3/Icechunk
    -   `run_all()`: Complete pipeline with cleanup

-   **`InputDatasetConfig`**: Pydantic configuration with environment variable support
    -   `OCR_INPUT_DATASET_S3_BUCKET`: S3 bucket (default: `carbonplan-ocr`)
    -   `OCR_INPUT_DATASET_S3_REGION`: AWS region (default: `us-west-2`)
    -   `OCR_INPUT_DATASET_BASE_PREFIX`: S3 prefix (default: `input/fire-risk`)
    -   `OCR_INPUT_DATASET_DEBUG`: Enable debug logging

### Storage Utilities

-   **`IcechunkWriter`**: Manages Icechunk repository creation and writing

    -   Automatic conflict resolution with retries
    -   Supports both S3 and local storage
    -   Dry-run mode for testing

-   **`S3Uploader`**: Handles S3 file uploads with progress tracking
    -   Single file and directory uploads
    -   Pattern-based file filtering
    -   Dry-run mode

## Adding a New Dataset

1. **Create a processor class** in `ocr/input_datasets/tensor/` or `ocr/input_datasets/vector/`:

```python
from ocr.input_datasets.base import BaseDatasetProcessor

class MyDatasetProcessor(BaseDatasetProcessor):
    dataset_name = 'my-dataset'
    dataset_type = 'tensor'
    description = 'My dataset description'

    # Dataset-specific configuration
    COILED_WORKERS = 10
    VARIABLES = {'var1': 'https://...', 'var2': 'https://...'}

    def download(self):
        # Download logic
        pass

    def process(self):
        # Processing logic
        pass
```

2. **Register in CLI** (`ocr/input_datasets/cli.py`):

```python
DATASET_REGISTRY = {
    'my-dataset': {
        'processor_class': MyDatasetProcessor,
        'type': 'tensor',
        'description': 'My dataset description',
    },
}
```

3. **Add tests** in `tests/test_input_datasets.py`

## Current Datasets

### scott-et-al-2024

USFS Wildfire Risk to Communities (2nd Edition, RDS-2020-0016-02)

-   **Type**: Tensor (raster)
-   **Variables**: BP, CRPS, CFL, Exposure, FLEP4, FLEP8, RPS, WHP
-   **Pipeline**:
    1. Download 8 TIFF files from USFS Box
    2. Merge TIFFs into Icechunk (EPSG:5070)
    3. Reproject to EPSG:4326

**Usage**:

```bash
# Download only
ocr ingest-data download scott-et-al-2024

# Process only (assumes data is downloaded)
ocr ingest-data process scott-et-al-2024

# Full pipeline
ocr ingest-data run-all scott-et-al-2024 --use-coiled
```

## CLI Reference

### Commands

-   **`list-datasets`**: Show all available datasets
-   **`download <dataset>`**: Download raw source data
-   **`process <dataset>`**: Process and upload to S3/Icechunk
-   **`run-all <dataset>`**: Complete pipeline (download + process + cleanup)

### Options

-   **`--dry-run`**: Preview operations without executing
-   **`--use-coiled`**: Use Coiled for distributed processing (where supported)
-   **`--debug`**: Enable debug logging

## Migration from `input-data/`

The old `input-data/` directory contained standalone scripts with significant code duplication. These have been migrated to the package for:

-   ✅ **Shared infrastructure**: Reusable base classes and utilities
-   ✅ **Testability**: Unit and integration tests
-   ✅ **Consistency**: Unified CLI interface
-   ✅ **Maintainability**: Single source of truth for common patterns

Old scripts in `input-data/` are deprecated and contain migration notices pointing to the new locations.
