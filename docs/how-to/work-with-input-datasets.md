# Working With Input Datasets (technical reference + how-to)

This page documents the main input datasets used in Open Climate Risk and shows how to access them programmatically via the `ocr` dataset catalog. Treat this as a technical reference for dataset names, example usage, and ingestion notes. Information about data provenance is available on the [Data Sources](../reference/data-sources.md) page.

Accessing the catalog

```python
from ocr import catalog

# List datasets
print(catalog)

# Load a dataset as an xarray / geopandas object
# rps_30 = catalog.get_dataset('USFS-wildfire-risk-communities').to_xarray()
```

## Tensor data (raster / Zarr)

These are n-dimensional raster datasets stored in Zarr/Icechunk stores.

### USFS Wildfire Risk to Communities

- Source: [Scott et al. 2024](../reference/data-sources.md#scott-et-al-2024)
- Ingested to: `input-data/tensor/USFS_fire_risk/`
- Typical usage:

```python
from ocr import catalog
rps_30 = catalog.get_dataset('USFS-wildfire-risk-communities').to_xarray()
```

### USFS climate runs (2011 / 2047)

- Source: [Riley et al. 2025](../reference/data-sources.md#riley-et-al-2025)
- These are stored as zipped archives that the ingestion scripts expand into Icechunk stores.

```python
climate_run_2011 = catalog.get_dataset('2011-climate-run-30m-4326').to_xarray()
climate_run_2047 = catalog.get_dataset('2047-climate-run-30m-4326').to_xarray()
```

### Wind datasets

- Source: [Rasmussen et al. 2023](../reference/data-sources.md#rasmussen-et-al-2023)
- Wind datasets and versions may change; if you add or switch wind sources, update the ingestion script under `input-data/` and register the new dataset with the `ocr` catalog.

## Vector data

Vector data are building footprints, administrative boundaries, and other GIS vector layers used for exposure and aggregation.

### Overture buildings

- Source: [Overture Maps Foundation buildings dataset](../reference/data-sources.md#overture-maps-foundation-buildings-dataset)
- Ingested subset for CONUS in `input-data/vector/overture_vector/`

```python
conus_buildings = catalog.get_dataset('conus-overture-buildings')
```

## Ingestion notes

- All ingestion scripts live in `input-data/`. When adding a new dataset:
    1. Add a script under `input-data/` that downloads, preprocesses, and writes data to an Icechunk store or geoparquet.
    2. Add a registration entry in the `ocr` catalog so `catalog.get_dataset(name)` returns a usable object.
    3. Add update [this page](/) as well as the [Data Sources](../reference/data-sources.md) page.

Contact the maintainers if you need access to private data buckets or credentials to download certain datasets.
