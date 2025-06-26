# Data Pipeline

![](../assets/ocr_data_flow.png)

## Deployment / Job Orchestration

`ocr/deploy.py` is a click CLI coordination / orchestration script. It will launch multiple parallel jobs of `01_Write_Region.py`, monitor the jobs with blocking behavior before dispatching `02_Pyramid (non-blocking)`, `02_Aggregate (blocking)` and `03_Tiles (blocking)`.

<!-- prettier-ignore-start -->
::: mkdocs-click
    :module: ocr.deploy
    :command: main
<!-- prettier-ignore-end -->

## 01_Write_Regions

1. `run_wind_region` - for a given region_id, applies wind adjustment and writes that region to an icechunk template.
2. `sample_risk_region` - for a given region_id, samples the building centroids to the wind adjusted risk raster to get building level risks and writes those to region_specific geoparquet files.

## 02_Pyramid

python script that runs ndpyramid coarsen to create a single level multi-scale Zarr pyramid. Default is non-reprojecting (EPSG:4326), but has a boolean flag for reprojecting to web-mercator `-r`. Note: reprojecting uses significantly more resources, so a larger machine may be required (rec r8g or r7a series for high ram).

## 02_Aggregate

python script that uses duckdb spatial to aggregate region_id specific geoparquet files into a single geoparquet file.

## 03_Tiles

bash script that uses GDAL and tippecanoe to create PMTiles from geoparquet.
TODO: Update this to a python deployment

# Example usage:

- run a single region: `uv run python deploy.py -r y2_x4`
- run multiple regions in parallel: `uv run python deploy.py -r y2_x4 -r y2_x5 -r y2_x6`
- run a single region on the prod branch: `uv run python deploy.py -b prod -r y2_x4`
- run a single region and generate pyramids: `uv run python deploy.py -r y2_x4 -p`
