# Input Dataset Processing

This directory contains the unified infrastructure for ingesting and processing input datasets for the OCR project.

## Documentation

For comprehensive documentation on using the input dataset ingestion system, see:

**[Input Dataset Ingestion Guide](../../docs/how-to/input-dataset-ingestion.md)**

## Quick Reference

List all available datasets:

```bash
ocr ingest-data list-datasets
```

Process a dataset:

```bash
ocr ingest-data run-all <dataset> --dry-run  # Preview first
ocr ingest-data run-all <dataset>            # Execute
```

## Architecture Overview

- **Base Classes**: `BaseDatasetProcessor`, `InputDatasetConfig`
- **Storage Utilities**: `IcechunkWriter`, `S3Uploader`
- **Tensor Datasets**: USFS fire risk data (scott-et-al-2024, riley-et-al-2025, dillon-et-al-2023)
- **Vector Datasets**: Overture Maps, Census TIGER/Line

## Developer Guide

See the [full documentation](../../docs/how-to/input-dataset-ingestion.md) for:

- Adding new datasets
- Using static methods for distributed processing
- Registering datasets in CLI
- Configuration options
- Troubleshooting

## Architecture

### Base Classes

- **`BaseDatasetProcessor`**: Abstract base class for all dataset processors
    - `download()`: Download raw source data
    - `process()`: Process and upload to S3/Icechunk
    - `run_all()`: Complete pipeline with cleanup

- **`InputDatasetConfig`**: Pydantic configuration with environment variable support
    - `OCR_INPUT_DATASET_S3_BUCKET`: S3 bucket (default: `carbonplan-ocr`)
    - `OCR_INPUT_DATASET_S3_REGION`: AWS region (default: `us-west-2`)
    - `OCR_INPUT_DATASET_BASE_PREFIX`: S3 prefix (default: `input/fire-risk`)
    - `OCR_INPUT_DATASET_DEBUG`: Enable debug logging

### Storage Utilities

- **`IcechunkWriter`**: Manages Icechunk repository creation and writing
    - Automatic conflict resolution with retries
    - Supports both S3 and local storage
    - Dry-run mode for testing

- **`S3Uploader`**: Handles S3 file uploads with progress tracking
    - Single file and directory uploads
    - Pattern-based file filtering
    - Dry-run mode

## Developer Guide

### Adding a New Dataset

1. **Create a processor class** in `ocr/input_datasets/tensor/` or `ocr/input_datasets/vector/`:

```python
from ocr.input_datasets.base import BaseDatasetProcessor, InputDatasetConfig

class MyDatasetProcessor(BaseDatasetProcessor):
    """Processor for My Dataset."""

    dataset_name: str = 'my-dataset'
    dataset_type = 'tensor'  # or 'vector'
    description: str = 'Brief description of the dataset'
    source_url: str = 'https://source.org/dataset'
    version: str = '2024'

    # Dataset-specific configuration (optional)
    COILED_WORKERS = 10
    VARIABLES = {'var1': 'https://...', 'var2': 'https://...'}

    def __init__(
        self,
        config: InputDatasetConfig | None = None,
        *,
        dry_run: bool = False,
        use_coiled: bool = False,  # if applicable
    ):
        super().__init__(config=config, dry_run=dry_run)
        self.use_coiled = use_coiled

    def download(self) -> None:
        """Download raw source data.

        Use self.retrieve() for downloading with pooch (caching + hash verification).
        For vector datasets that query S3 directly, this can be a no-op.
        """
        console.log('Downloading...')
        # Implementation

    def process(self) -> None:
        """Process and upload data to S3/Icechunk.

        Use static methods for processing logic that can be distributed.
        """
        console.log(f'Processing {self.dataset_name}...')
        # Implementation
```

2. **Use static methods for distributed processing**:

Static methods can be serialized and sent to Coiled workers:

```python
@staticmethod
def _process_chunk(
    input_path: str,
    output_path: str,
    dry_run: bool = False,
) -> None:
    """Process a single chunk (can be distributed)."""
    if dry_run:
        console.log(f'[DRY RUN] Would process {input_path}')
        return
    # Actual processing logic
```

3. **Register in CLI** (`ocr/input_datasets/cli.py`):

```python
from ocr.input_datasets.vector.my_dataset import MyDatasetProcessor

DATASET_REGISTRY = {
    'my-dataset': {
        'processor_class': MyDatasetProcessor,
        'type': 'vector',
        'description': 'Brief description matching processor',
    },
}
```

4. **Add dataset-specific CLI options** (if needed):

For custom parameters like `--overture-data-type`, update both `process` and `run_all` commands in `cli.py`.
