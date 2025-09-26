# API reference

This page provides a structured, auto-generated reference for the OCR Python package using mkdocstrings. Each section links to the corresponding module(s) and surfaces docstrings, type hints, and signatures.

---

## Package overview

High-level package entry points and public exports.

::: ocr

---

## Core modules

### Configuration and types

- Configuration models for storage, chunking, coiled, and processing settings
- Strongly typed enums for environment, platform, and risk types

::: ocr.config

::: ocr.types

### Data access and utilities

- Dataset and Catalog abstractions for Zarr and GeoParquet on S3/local
- Helpers for DuckDB (extension loading, S3 secrets), vector sampling, and file transfer

::: ocr.datasets

::: ocr.utils

### CONUS404 helpers

- Load CONUS404 variables, compute relative humidity, wind rotation and diagnostics
- Geographic selection utilities (point/bbox) with CRS-aware transforms

::: ocr.conus404

---

## Risk utilities

Core fire/wind risk utilities used by the pipeline (kernels, wind classification, risk composition).

::: ocr.risks.fire

---

## Deployment and CLI

### Batch managers

Orchestration backends for local and Coiled execution.

::: ocr.deploy.managers

### CLI application (Typer)

Command-line interface exposed as the `ocr` command. For detailed usage and options, see Tutorials as well.

<!-- prettier-ignore-start -->
::: mkdocs-click
  :module: ocr.deploy.cli
  :command: ocr
  :prog_name: ocr
  :list_subcommands: true
<!-- prettier-ignore-end -->

---
