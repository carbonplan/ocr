# API reference

This page provides a structured, auto-generated reference for the `ocr` Python package using mkdocstrings. Each section links to the corresponding module(s) and surfaces docstrings, type hints, and signatures.

---

## Package overview

High-level package entry points and public exports.

::: ocr

---

## Core modules

### Configuration

Configuration models for storage, chunking, Coiled, and processing settings.

::: ocr.config

### Type definitions

Strongly typed enums for environment, platform, and risk types.

::: ocr.types

---

## Data access

### Datasets

Dataset and Catalog abstractions for Zarr and GeoParquet on S3/local storage.

::: ocr.datasets

### CONUS404 helpers

Load CONUS404 variables, compute relative humidity, wind rotation and diagnostics. Geographic selection utilities (point/bbox) with CRS-aware transforms.

::: ocr.conus404

---

## Utilities

### General utilities

Helpers for DuckDB (extension loading, S3 secrets), vector sampling, and file transfer.

::: ocr.utils

### Testing utilities

Snapshot testing extensions for xarray and GeoPandas.

::: ocr.testing

---

## Risk analysis

### Fire risk

Core fire/wind risk utilities used by the pipeline (kernels, wind classification, risk composition).

::: ocr.risks.fire

---

## Internal pipeline modules

:::{note}
**Internal API**

These modules are used internally by the pipeline and are not intended for direct public consumption. They are documented here for completeness and advanced use cases.
:::

### Batch managers

Orchestration backends for local and Coiled execution.

::: ocr.deploy.managers

### CLI application

Command-line interface exposed as the `ocr` command. For detailed usage and options, see the [Data Pipeline](../how-to/data-pipeline.md) guide.

<!-- prettier-ignore-start -->
::: mkdocs-click
  :module: ocr.deploy.cli
  :command: ocr
  :prog_name: ocr
  :list_subcommands: true
<!-- prettier-ignore-end -->
