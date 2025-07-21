# Data Pipeline

![](../assets/ocr_data_flow.png)

## Deployment / Job Orchestration

`ocr/deploy.py` is a click CLI coordination / orchestration script. It will launch multiple parallel jobs of `01_Write_Region.py`, monitor the jobs with blocking behavior before dispatching `02_Aggregate (blocking)` and `03_Tiles (blocking)`. You can monitor jobs in the [coiled dashboard](https://cloud.coiled.io/clusters?workspace=carbonplan).

<!-- prettier-ignore-start -->
::: mkdocs-click
    :module: ocr.deploy
    :command: main
<!-- prettier-ignore-end -->

## 01_Write_Region.py

`run_wind_region`

For a given region_id, applies wind adjustment and writes that region to an icechunk template.

`sample_risk_region`

For a given region_id, samples the building centroids to the wind adjusted risk raster to get building level risks and writes those to region_specific geoparquet files.

<!--
## 02_Pyramid

python script that runs ndpyramid coarsen to create a single level multi-scale Zarr pyramid. Default is non-reprojecting (EPSG:4326), but has a boolean flag for reprojecting to web-mercator `-r`. Note: reprojecting uses significantly more resources, so a larger machine may be required (rec r8g or r7a series for high ram). -->

## 02_Aggregate.py

Python script that uses `duckdb` spatial to aggregate region_id specific geoparquet files into a single geoparquet file.

## 03_Tiles.sh

Bash script that uses `GDAL` and `tippecanoe` to create PMTiles from aggregated geoparquet.

## Choosing valid `region_id`'s

Our input 30 meter dataset and chunking schema contains multiple empty regions. To avoid running regions with no source data, there are some helper utilities to see the available `valid` region_ids.

```python

from ocr.chunking_config import ChunkingConfig
config = ChunkingConfig()

config.valid_region_ids
```

Should return a list of valid `region_ids`. ex: `['y1_x3', 'y1_x4', ...]`.

# Example usage:

- run a single region: `uv run python deploy.py -r y2_x4`
- run multiple regions in parallel: `uv run python deploy.py -r y2_x4 -r y2_x5 -r y2_x6`
- run a single region on the prod branch with the wipe flag enabled: `uv run python deploy.py -w -b prod -r y2_x4`

## Running CONUS

Running this command should return a CLI string containing all of the valid region_ids. You should check the other flags and alter them as needed.

```python

from ocr.chunking_config import ChunkingConfig
config = ChunkingConfig()

config.generate_click_all_region_deploy_command
```

## Usage notes:

**-b, --branch**

- The default for this flag is `QA`.
- `QA`: The Icechunk store will be wiped as well as the individual geoparquet region_id files.
- `prod`: The Icechunk store will be appended to and new geoparquet region_id files will be written.

**-w, --wipe**

- Wipes the Icechunk store and re-initializes it.
- Wipes any region_id geoparquet files.

**-r, --region**

- for multiple regions, you must supply a `-r` flag for each one. ex: `-r y2_x4 -r y3_x4`.
- `region(s)` must be in the format `y{n}_x{n}` and correspond to valid region_ids in the Icechunk template.
- You can visualize the available regions:

**-s, --summary-stats**

- The default for this flag is `False`.
- If enabled, this will create a county level summary stats geoparquet and PMTiles.

```python
from ocr.chunking_config import ChunkingConfig

config = ChunkingConfig()
config.plot_all_chunks()
```
