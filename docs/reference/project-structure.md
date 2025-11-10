# Project structure

This page documents the OCR repository layout and explains the purpose of key directories and files. Use this as a technical reference when contributing code, adding datasets, or extending documentation.

---

## Repository overview

The OCR platform is organized into distinct layers: the core Python package (`ocr/`), supporting infrastructure (configuration, deployment, testing), input data management, exploratory research notebooks, and comprehensive documentation. The structure follows best practices for scientific Python projects with emphasis on reproducibility, modularity, and cloud-native execution.

```mermaid
graph TB
    subgraph Repository["OCR Repository"]
        subgraph Core["Core Package"]
            OCR[ocr/]
            OCR --> CONFIG[config.py - Configuration models]
            OCR --> TYPES[types.py - Type definitions]
            OCR --> DATASETS[datasets.py - Data catalog]
            OCR --> CONUS[conus404.py - Climate data]
            OCR --> UTILS[utils.py - Utilities]
            OCR --> TESTING[testing.py - Test helpers]

            OCR --> DEPLOY[deploy/]
            DEPLOY --> CLI[cli.py - CLI app]
            DEPLOY --> MANAGERS[managers.py - Orchestration]

            OCR --> PIPELINE[pipeline/]
            PIPELINE --> PROCESS[process_region.py]
            PIPELINE --> PARTITION[partition.py]
            PIPELINE --> STATS[fire_wind_risk_regional_aggregator.py]
            PIPELINE --> PMTILES[create_building_pmtiles.py + create_regional_pmtiles.py]
            PIPELINE --> WRITERS[write_*_analysis_files.py]

            OCR --> RISKS[risks/]
            RISKS --> FIRE[fire.py - Risk models]
        end

        subgraph Data["Data & Inputs"]
            INPUT[input-data/]
            INPUT --> TENSOR[tensor/ - CONUS404, USFS fire risk]
            INPUT --> VECTOR[vector/ - Buildings, regions, structures]
        end

        subgraph Research["Research & Exploration"]
            NOTEBOOKS[notebooks/]
            NOTEBOOKS --> NB1[Wind analysis notebooks]
            NOTEBOOKS --> NB2[Fire risk kernels]
            NOTEBOOKS --> NB3[Scaling experiments]
        end

        subgraph Docs["Documentation"]
            DOCSDIR[docs/]
            DOCSDIR --> HOWTO[how-to/ - Guides]
            DOCSDIR --> METHODS[methods/ - Science docs]
            DOCSDIR --> REFERENCE[reference/ - API & specs]
            DOCSDIR --> TUTORIALS[tutorials/ - Workflows]
        end

        subgraph Testing["Testing & QA"]
            TESTS[tests/]
            TESTS --> UNIT[Unit tests]
            TESTS --> INTEGRATION[Integration tests]
            TESTS --> SNAPSHOTS[Snapshot tests]
        end

        subgraph Config["Configuration & Build"]
            PYPROJECT[pyproject.toml - Package config]
            PIXI[pixi.lock - Environment lock]
            MKDOCS[mkdocs.yml - Docs config]
            ENV[ocr-*.env - Environment vars]
            GITHUB[.github/ - CI/CD workflows]
        end

        subgraph Infra["Infrastructure"]
            BUCKET[bucket_creation/ - S3 setup]
        end
    end

    style Core fill:#e1f5ff
    style Data fill:#fff4e1
    style Research fill:#f3e8ff
    style Docs fill:#e8f5e9
    style Testing fill:#ffebee
    style Config fill:#fafafa
    style Infra fill:#fff9c4
```

---

## Core package (`ocr/`)

Contains all production code organized into logical modules:

### Top-level modules

| Module        | Purpose                                                                               |
| ------------- | ------------------------------------------------------------------------------------- |
| `config.py`   | Pydantic models for storage, chunking, Coiled, and processing configuration           |
| `types.py`    | Type definitions and enums (Environment, Platform, RiskType, RegionType)              |
| `datasets.py` | Catalog abstraction for Zarr and GeoParquet datasets in S3 storage                    |
| `conus404.py` | CONUS404 climate data helpers: load variables, compute humidity, wind transformations |
| `utils.py`    | DuckDB utilities, S3 secrets, vector sampling, file transfer helpers                  |
| `testing.py`  | Snapshot testing extensions for xarray and GeoPandas                                  |
| `console.py`  | Rich console instance for pretty terminal output                                      |

### Deployment (`deploy/`)

Orchestration layer for local and cloud execution:

- **`cli.py`** - Typer-based CLI application (`ocr` command) with commands for processing regions, aggregation, PMTiles generation, and analysis file creation
- **`managers.py`** - Abstract batch manager interface with `CoiledBatchManager` (cloud) and `LocalBatchManager` (local) implementations

### Pipeline (`pipeline/`)

Internal processing modules coordinated by the CLI. These implement the data processing workflow:

- **`process_region.py`** - Sample risk values to building locations
- **`partition.py`** - Partition GeoParquet by geographic regions
- **`fire_wind_risk_regional_aggregator.py`** - Compute regional statistics with DuckDB
- **`create_building_pmtiles.py`** - Generate PMTiles for building data visualization
- **`create_regional_pmtiles.py`** - Generate region-specific PMTiles
- **`write_aggregated_region_analysis_files.py`** - Write regional summary tables
- **`write_per_region_analysis_files.py`** - Write per-region analysis outputs (CSV/JSON)

### Risk models (`risks/`)

Domain-specific risk calculation logic:

- **`fire.py`** - Fire/wind risk kernels, wind classification, elliptical spread models

---

## Data management (`input-data/`)

Organized storage for input datasets and ingestion scripts:

### Tensor data (`tensor/`)

- **`conus404/`** - CONUS404 climate reanalysis data
- **`USFS_fire_risk/`** - USFS fire risk rasters

### Vector data (`vector/`)

- **`aggregated_regions/`** - Regional boundaries for aggregation
- **`alexandre-2016/`** - Historical fire perimeter data
- **`calfire_stuctures_destroyed/`** - Structure damage records
- **`overture_vector/`** - Building footprints from Overture Maps

!!! note

    Raw data files are typically not committed. This directory contains ingestion scripts and metadata. Large datasets are stored on S3.

---

## Research notebooks (`notebooks/`)

Exploratory Jupyter notebooks for prototyping and analysis:

- `conus404-winds.ipynb` - Wind data exploration
- `elliptical_kernel.ipynb` - Fire spread kernel development
- `fire-weather-wind-mode-reprojected.ipynb` - Wind mode analysis
- `horizontal-scaling-via-spatial-chunking.ipynb` - Performance optimization experiments
- `wind_spread.ipynb` - Wind-driven fire spread modeling

!!! note

    **Convention**: When a notebook reaches maturity and demonstrates stable workflows, consider converting it into a tutorial under `docs/tutorials/`.

---

## Documentation (`docs/`)

Follows the [Diátaxis framework](https://diataxis.fr/) for structured documentation:

```text
docs/
├── how-to/              # Task-oriented guides
│   ├── installation.md
│   ├── getting-started.md
│   ├── snapshot-testing.md
│   ├── release-procedure.md
│   └── work-with-data.ipynb
├── tutorials/           # Learning-oriented lessons
│   ├── calculate-wind-adjusted-fire-risk-for-a-region.ipynb
│   └── data-pipeline.md
├── reference/           # Information-oriented technical specs
│   ├── api.md           # Auto-generated API docs
│   ├── data-schema.md
│   ├── data-downloads.md
│   ├── deployment.md
│   └── project-structure.md (this file)
├── methods/             # Explanation-oriented background
│   └── fire-risk/       # Scientific methodology
├── assets/              # Images, stylesheets, static files
└── index.md             # Documentation home page
```

Documentation is built with MkDocs Material and deployed automatically on merge to `main`.

---

## Testing (`tests/`)

Comprehensive test suite with unit and integration tests:

| File                         | Purpose                                               |
| ---------------------------- | ----------------------------------------------------- |
| `conftest.py`                | Pytest fixtures and configuration                     |
| `test_config.py`             | Configuration model validation                        |
| `test_conus404.py`           | CONUS404 data loading and transformations             |
| `test_datasets.py`           | Dataset catalog and access patterns                   |
| `test_managers.py`           | Batch manager orchestration logic                     |
| `test_utils.py`              | Utility function tests                                |
| `test_pipeline_snapshots.py` | Snapshot-based integration tests for pipeline outputs |
| `risks/`                     | Risk model tests                                      |

**Test execution:**

```bash
pixi run tests              # Unit tests only
pixi run tests-integration  # Integration tests (may require S3 access)
```

---

## Configuration files

### Package and environment

- **`pyproject.toml`** - Project metadata, dependencies (managed by Pixi), build config, tool settings (ruff, pytest, coverage)
- **`pixi.lock`** - Locked dependency versions for reproducible environments
- **`environment.yaml`** - Conda environment export (auto-generated from Pixi for Coiled deployments)

### Documentation

- **`mkdocs.yml`** - MkDocs configuration: theme, plugins, navigation structure

### Environment templates

- **`ocr-local.env`** - Template for local development (uses local filesystem)
- **`ocr-coiled-s3.env`** - Template for cloud execution (S3 backend)
- **`ocr-coiled-s3-staging.env`** - Staging environment configuration
- **`ocr-coiled-s3-production.env`** - Production environment configuration

### Code quality

- **`.pre-commit-config.yaml`** - Pre-commit hooks for linting and formatting
- **`.prettierrc.json`** - Prettier configuration for Markdown/YAML formatting
- **`codecov.yml`** - Code coverage reporting configuration

---

## Infrastructure (`bucket_creation/`)

Helper scripts for cloud infrastructure setup:

- **`create_s3_bucket.py`** - Script to create and configure S3 buckets with appropriate permissions and lifecycle policies

---

## CI/CD (`.github/`)

GitHub Actions workflows for automated testing, building, and deployment:

- **`workflows/`** - CI/CD pipeline definitions (tests, linting, docs deployment, releases)
- **`scripts/`** - Helper scripts for environment export and Coiled software creation
- **`copilot-instructions.md`** - Project-specific instructions for GitHub Copilot
- **`dependabot.yaml`** - Automated dependency updates configuration
- **`release-drafter.yml`** - Automated release notes generation

---

## Development workflows

### Adding new code

1. **Create module** under `ocr/` (or in appropriate subpackage)
2. **Add tests** under `tests/` (unit tests are required, integration tests for complex scenarios)
3. **Update documentation**:
    - Add how-to guide if introducing new user-facing workflow
    - Update API reference if adding public functions/classes
    - Add method explanation if introducing new scientific approach

### Adding new datasets

1. **Create ingestion script** under `input-data/tensor/` or `input-data/vector/`
2. **Register dataset** in `ocr.datasets` catalog with metadata
3. **Document provenance** under `docs/methods/` with sources, processing steps, and limitations
4. **Add schema documentation** to `docs/reference/data-schema.md`

### Updating documentation

1. **Choose appropriate section** based on Diátaxis framework:
    - How-to guides: task-oriented, assume prior knowledge
    - Tutorials: learning-oriented, step-by-step for beginners
    - Reference: information-oriented, technical specifications
    - Methods: explanation-oriented, scientific background
2. **Update navigation** in `mkdocs.yml` if adding new pages
3. **Test locally**: `pixi run docs-serve` to preview changes
4. **Submit PR**: Documentation builds are tested in CI

### Release workflow

See `docs/how-to/release-procedure.md` for detailed release instructions.
