# Data Pipeline

![](../assets/ocr_data_flow.png)

## main

#### `region_id` based atomic unit jobs

`ocr/main.py` is a click CLI that can be used to dispatch parallel batch jobs of `01_write_region`.

<!-- prettier-ignore-start -->
::: mkdocs-click
    :module: ocr.main
    :command: main
<!-- prettier-ignore-end -->

## 01_Write_Regions

1. `run_wind_region` - for a given region_id, applies wind adjustment and writes that region to an icechunk template.
2. `sample_risk_region` - for a given region_id, samples the building centroids to the wind adjusted risk raster to get building level risks and writes those to region_specific geoparquet files.

## 02_Aggregate

python script that uses duckdb spatial to aggregate region_id specific geoparquet files into a single geoparquet file.

## 03_Tiles

bash script that uses GDAL and tippecanoe to create PMTiles from geoparquet.

# Example usage:

- run a single region on coiled batch: `uv run python main.py -c -r y2_x4`
- run multiple regions on coiled batch in parallel: `uv run python main.py -c -r y2_x4 -r y2_x5 -r y2_x6`
- run a single region on prod branch on coiled batch: `uv run python main.py -c -b prod -r y2_x4`
