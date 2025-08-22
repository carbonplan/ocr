# Installation

This guide shows two quick ways to install OCR for development and for a simple pip-based install.

## Prerequisites

- Python 3.12+ installed
- Git
- Optional: `pixi` for reproducible developer environments (recommended)

## Option A — Recommended (pixi)

1. Install pixi (see [pixi](https://pixi.sh)):

```bash
# macOS / Linux
curl -sSf https://sh.pixi.sh | sh
```

1. From the repo root, install dependencies and start a dev shell:

```bash
pixi install
pixi shell
```

1. Inside the pixi shell, run tests or development commands:

```bash
pixi run pytest
pixi run format
```

## Option B — pip (quick install)

```bash
pip install git+https://github.com/carbonplan/ocr
```

Note: pip install is useful for quickly installing the package, but the pixi workflow is recommended for development because it creates a reproducible environment and includes development-only dependencies.

## Environment configuration

Copy an example env file and edit values:

```bash
cp ocr-local.env .env
# edit .env to set OCR_STORAGE_ROOT, OCR_ENVIRONMENT, OCR_DEBUG, and any credentials
```

Key variables to set in `.env`:

- `OCR_STORAGE_ROOT` — S3 path or local path where outputs are written (e.g. `s3://your-bucket/`).
- `OCR_ENVIRONMENT` — name of the environment (e.g. `QA`,  `STAGING`, `PROD`).
- `OCR_DEBUG` — `1` to enable verbose logging.

## Verification

Run a quick smoke test to ensure the package imports and CLI are available:

```bash
python -c "import ocr; print('ocr', ocr.__version__)"
ocr --help
```
