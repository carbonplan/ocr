# TODO

- Qualitative description of methods
- Link out to more detailed pages (data sources, region atomization, etc.)

This page explains the high-level method used to compute building-level fire risk scores. It is an explanation-oriented document (conceptual + references) rather than a step-by-step tutorial.

## Summary

- The model combines spatial datasets describing wildfire hazard, exposure (building footprints), and vulnerability (damageability) to compute a per-building risk score.
- Risk is computed per-region (region IDs follow the y{tile}\_x{tile} tiling used across the codebase) and aggregated into geoparquet outputs.

## Inputs

- Raster hazard layers (USFS wildfire risk products, wind fields) stored in Zarr/Icechunk stores.
- Building footprints (Overture) as vector geoparquet.

## Processing steps (high level)

## Outputs

- Regional and aggregated geoparquet files containing building-level risk metrics and metadata.
- An aggregated 1, 15 and 30 year time-horizon regional summary statistics PMTiles file.
- Summary stats files (geoparquet, geojson, csv) for region (county / census tract) aggregations.

## References and further reading

- USFS Wildfire Risk datasets (see [explanations/input-datasets.md](../explanations/input-datasets.md) for access details)

- Relevant academic literature on vulnerability functions and exposure modeling (bibliography to add as needed).

If you need a runnable step-by-step tutorial for reproducing the calculations locally, see the [tutorials/data-pipeline.md](../tutorials/data-pipeline.md) tutorial and run the `ocr process-region` command with `OCR_DEBUG=1` to inspect intermediate artifacts.
