# Install OCR for Development

This guide shows how to set up OCR for local development using pixi.

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
git clone https://github.com/carbonplan/ocr.git
cd ocr
pixi install
```

### 3. Configure environment

```bash
cp ocr-local.env .env
```

Edit `.env` and set required variables:

- `OCR_STORAGE_ROOT` — where outputs are written (e.g., `s3://your-bucket/` or local path)
- `OCR_ENVIRONMENT` — environment name (`QA`, `STAGING`, or `PROD`)
- `OCR_DEBUG` — set to `1` for verbose logging

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

- Run [pre-commit checks](../reference/project-structure.md#pre-commit): `pre-commit run --all-files`
- Run [integration tests with snapshots](snapshot-testing.md): `pixi run tests -m integration`
- See [getting started](getting-started.md) to begin using OCR
