# Data Downloads

OCR provides a few different ways to access the data, in addition to exploration via the web tool. This page outlines the different versions and formats of data available for download. To understand the contents of each download type, read more about the [data schema](./data-schema.md).

## Download options

| Option                            | Description                                                                         | Formats                           | Access               |
| --------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------- | -------------------- |
| **Raster (tensor) data**          | Full gridded dataset spanning CONUS                                                 | Icechunk (Zarr-based)             | S3                   |
| **Vector (point) data**           | Full buildings dataset spanning CONUS                                               | GeoParquet (schema version 1.1.0) | S3                   |
| **Regional statistics**           | Summary statistics for regions (county, census tract, census block) within CONUS    | CSV, GeoJSON                      | S3                   |
| **Subsetted vector (point) data** | Building-level data subsetted to active region (county, census tract, census block) | CSV, GeoPackage                   | Web tool (see below) |

## Download links

### Full dataset downloads

| Option                   | Path                                                                                         | Notes                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Raster (tensor) data** | `s3://carbonplan-ocr/output/fire-risk/tensor/production/v0.8.0/ocr.icechunk/`                | See [guide for working with Icehunk data](../how-to/work-with-data.ipynb)    |
| **Vector (point) data**  | `s3://carbonplan-ocr/output/fire-risk/vector/production/v0.8.0/geoparquet/buildings.parquet` | See [guide for working with GeoParquet data](../how-to/work-with-data.ipynb) |

TODO: update/finalize version pointers

### Regional statistics downloads

|                                     | CSV                                                                                                      | GeoJSON                                                                                                          |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Statistics across **counties**      | [`stats.csv`](s3://carbonplan-ocr/output/fire-risk/production/v0.8.0/region-analysis/counties/stats.csv) | [`stats.geojson`](s3://carbonplan-ocr/output/fire-risk/production/v0.8.0/region-analysis/counties/stats.geojson) |
| Statistics across **census tracts** | [`stats.csv`](s3://carbonplan-ocr/output/fire-risk/production/v0.8.0/region-analysis/tracts/stats.csv)   | [`stats.geojson`](s3://carbonplan-ocr/output/fire-risk/production/v0.8.0/region-analysis/tracts/stats.geojson)   |
| Statistics across **census blocks** | [`stats.csv`](s3://carbonplan-ocr/output/fire-risk/production/v0.8.0/region-analysis/block/stats.csv)    | [`stats.geojson`](s3://carbonplan-ocr/output/fire-risk/production/v0.8.0/region-analysis/block/stats.geojson)    |

TODO: update/finalize version pointers

## Downloading subsetted data in the web tool

The [web tool](https://ocr.carbonplan.org/) can be used to access region-specific, subsetted downloads with the following steps:

1. Using the map or search bar, navigate to region of interest.
2. Scroll to the `Risk in the region` section in the sidebar.
3. Select your region of interest (county, census tract, or census block) and view trends inline.
4. Click `CSV ↓` or `GeoPackage ↓` to download building-level data for the selected region.

![](../assets/web-data-downloads.png)

TODO: update/finalize screenshot
