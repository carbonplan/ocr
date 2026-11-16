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

### Region grid

The Scott 2024 30 m CONUS grid split into regions, the parts fire `score` writes.

```{eval-rst}
.. automodule:: ocr.risks.fire.grid
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

Load CONUS404 variables, compute relative humidity, wind rotation, wind speed and direction.

```{eval-rst}
.. automodule:: ocr.risks.fire.conus404
   :members:
   :show-inheritance:
```

---

## Utilities

### General utilities

Geographic selection (point/bbox) with CRS-aware transforms.

```{eval-rst}
.. automodule:: ocr.risks.fire.utils
   :members:
   :show-inheritance:
```

### Testing utilities

Snapshot testing extension for xarray.

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
.. automodule:: ocr.risks.fire.model
   :members:
   :show-inheritance:
```

---

## Internal pipeline modules

:::{admonition} **Internal API**
:class: note

These modules are used internally by the pipeline and are not intended for direct public consumption. They are documented here for completeness and advanced use cases.
:::

### Pipeline stages

Stages, preflight checks and stage kwargs behind the `ocr` command. Usage: [Data pipeline](../how-to/data-pipeline.md).

```{eval-rst}
.. automodule:: ocr.stages
   :members:
```

### CLI application

The `ocr` command (`ocr run`, `ocr stage`), run files (`RunFile`) and Coiled job settings (`JobConfig`, `JOBS`).

```{eval-rst}
.. automodule:: ocr.cli
   :members:
```
