# Project layout

This page documents the repository layout at a glance and explains the purpose of the main top-level directories and files. Use this as a technical reference when adding modules, data ingestion scripts, docs or tests.

Top-level entries (short descriptions)

- `bucket_creation/` — helper script(s) to create and configure S3 buckets used by OCR; contains `create_s3_bucket.py`.
- `contributing.md` — contribution guidelines and PR checks.
- `design-docs/` — design notes and longer-form architecture documents.
- `docs/` — user and developer documentation (this is the MkDocs source):
- `input-data/` — ingestion scripts and local copies of small datasets; organized into `tensor/` and `vector/`.
- `intermediate/` — intermediate outputs used in staging/testing (e.g., `fire-risk/`).
- `notebooks/` — exploratory notebooks and reproducible analyses; consider converting stable notebooks into docs/tutorials when ready.
- `ocr/` — the Python package containing the CLI, pipeline logic, dataset catalog, and risk calculation modules. Key modules include `deploy` (CLI), `pipeline` (processing), `datasets` (catalog), and `risks` (risk models).
- `ocr-*.env` files — example environment files for local and cloud runs (`ocr-local.env`, `ocr-coiled-s3.env`).
- `pyproject.toml`, `pixi.lock` — packaging and environment metadata.
- `mkdocs.yml` — MkDocs configuration used to build the documentation site.
- `tests/` — unit and integration tests; add tests for new public behavior here.

---

- When adding code: add modules under `ocr/`, include tests under `tests/`, and update docs under `docs/` with a How-to and a Technical reference for any new public API.
- When adding datasets: add ingestion scripts under `input-data/`, register the dataset with `ocr.datasets` catalog, and document provenance in `docs/methods/`.
- For docs changes: update the MkDocs nav in `mkdocs.yml` if you create new sections or move files.
