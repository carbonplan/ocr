# API reference

This page provides a structured, auto-generated reference for the `ocr` Python package. Each section links to the corresponding module(s) and surfaces docstrings, type hints, and signatures.

---

## Package overview

High-level package entry points and public exports.

```{eval-rst}
.. automodule:: ocr
   :members:
   :show-inheritance:
```

---

## Core modules

### Configuration

Configuration models for storage, chunking, Coiled, and processing settings.

```{eval-rst}
.. automodule:: ocr.risks.fire.config
   :members:
   :show-inheritance:
```

### Type definitions

Strongly typed enums for environment, platform, and risk types.

```{eval-rst}
.. automodule:: ocr.risks.fire.types
   :members:
   :show-inheritance:
```

---

## Data access

### Datasets

Dataset and Catalog abstractions for Zarr and GeoParquet on S3/local storage.

```{eval-rst}
.. automodule:: ocr.risks.fire.datasets
   :members:
   :show-inheritance:
```

### CONUS404 helpers

Load CONUS404 variables, compute relative humidity, wind rotation and diagnostics. Geographic selection utilities (point/bbox) with CRS-aware transforms.

```{eval-rst}
.. automodule:: ocr.risks.fire.conus404
   :members:
   :show-inheritance:
```

---

## Utilities

### General utilities

Helpers for DuckDB (extension loading, S3 secrets), vector sampling, and file transfer.

```{eval-rst}
.. automodule:: ocr.risks.fire.utils
   :members:
   :show-inheritance:
```

### Testing utilities

Snapshot testing extensions for xarray and GeoPandas.

```{eval-rst}
.. automodule:: ocr.risks.fire.testing
   :members:
   :show-inheritance:
```

---

## Risk analysis

### Fire risk

Core fire/wind risk utilities used by the pipeline (kernels, wind classification, risk composition).

```{eval-rst}
.. automodule:: ocr.risks.fire.risks.fire
   :members:
   :show-inheritance:
```

---

## Internal pipeline modules

:::{admonition} **Internal API**
:class: note

These modules are used internally by the pipeline and are not intended for direct public consumption. They are documented here for completeness and advanced use cases.
:::

### Batch managers

Orchestration backends for local and Coiled execution.

```{eval-rst}
.. automodule:: ocr.risks.fire.deploy.managers
   :members:
   :show-inheritance:
```

### CLI application

Command-line interface exposed as the `ocr` command. For detailed usage and options, see the [Data Pipeline](../how-to/data-pipeline.md) guide.

```{eval-rst}
.. click:: ocr.risks.fire.deploy.cli:ocr
   :prog: ocr
   :nested: full
```
