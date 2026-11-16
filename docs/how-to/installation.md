# Install `ocr` for development

This guide shows how to set up Open Climate Risk (`ocr`) for local development using pixi.

## Prerequisites

- Python 3.12+
- Git

## Steps

### 1. Install pixi

```bash
# macOS / Linux
curl -sSf https://sh.pixi.sh | sh
```

For other platforms, see [pixi.sh](https://pixi.sh).

### 2. Clone and install dependencies

```bash
git clone https://github.com/carbonplan/ocr-multi-hazard.git
cd ocr-multi-hazard
pixi install
```

### 3. Configure environment

Storage is read from environment variables (`ocr.config.StorageConfig`); all are optional:

- `OCR_STORAGE_BUCKET` — S3 bucket for inputs and outputs (default `carbonplan-ocr`)
- `OCR_STORAGE_REGION` — its AWS region (default `us-west-2`)
- `OCR_STORAGE_LOCAL_ROOT` — write outputs under this local directory instead of S3, e.g. `~/ocr-data`

### 4. Verify installation

```bash
pixi run python -c "import ocr; print('ocr', ocr.__version__)"
pixi run ocr --help
```

### 5. Run tests

```bash
pixi run tests
```

## Next steps

- See [Data pipeline](data-pipeline.md) to run the processing pipeline
- Read [Project structure](../reference/project-structure.md) to understand the codebase
