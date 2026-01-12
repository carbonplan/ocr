# Data Downloads (Beta)

While OCR is in beta, there a few different ways to access the data, in addition to exploration via the web tool. This page outlines the different versions and formats of data available for download.

<!-- prettier-ignore-start -->
!!! warning "License Agreement"
  Open Climate Risk data are made available under the licenses listed below. By accessing the data, you agree to adhere to the license terms.

<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
!!! warning "Terms of Data Use"
By viewing Open Climate Risk data, you agree to the [Terms of Data Access](terms-of-data-access.md).

<!-- prettier-ignore-end -->

## Download options

| Option                            | Description                                                                         | Formats               | Access                                                       | License                                                   |
| --------------------------------- | ----------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| **Raster (tensor) data**          | Full gridded dataset spanning CONUS                                                 | Icechunk (Zarr-based) | [Source Coop](https://source.coop/carbonplan/carbonplan-ocr) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Vector (point) data**           | Full buildings dataset spanning CONUS                                               | GeoParquet            | [Source Coop](https://source.coop/carbonplan/carbonplan-ocr) | [ODbL](https://opendatacommons.org/licenses/odbl/1-0/)    |
| **Regional statistics**           | Summary statistics for regions (county, census tract, census block) within CONUS    | CSV, GeoJSON          | [Source Coop](https://source.coop/carbonplan/carbonplan-ocr) | [ODbL](https://opendatacommons.org/licenses/odbl/1-0/)    |
| **Subsetted vector (point) data** | Building-level data subsetted to active region (county, census tract, census block) | CSV, GeoPackage       | Web tool (see below)                                         | [ODbL](https://opendatacommons.org/licenses/odbl/1-0/)    |

## Full dataset downloads

### Links

| Option                   | Path                                                                                                                                    | Notes                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Raster (tensor) data** | `s3://us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/tensor/production/v0.13.1/ocr.icechunk/`                | See [guide for working with Icehunk data](./how-to/work-with-data.ipynb)    |
| **Vector (point) data**  | `s3://us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/geoparquet/buildings.parquet` | See [guide for working with GeoParquet data](./how-to/work-with-data.ipynb) |

### Schema

The schemas for each of the full datasets are described on the [data schema](./data-schema.md) page.

## Regional statistics downloads

### Links

|                                     | CSV                                                                                                                                                                                      | GeoJSON                                                                                                                                                                                          |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Statistics across **counties**      | [`stats.csv`](https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/region-analysis/counties/stats.csv) | [`stats.geojson`](https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/region-analysis/counties/stats.geojson) |
| Statistics across **census tracts** | [`stats.csv`](https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/region-analysis/tracts/stats.csv)   | [`stats.geojson`](https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/region-analysis/tracts/stats.geojson)   |
| Statistics across **census blocks** | [`stats.csv`](https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/region-analysis/block/stats.csv)    | [`stats.geojson`](https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/region-analysis/block/stats.geojson)    |
| Statistics across **states**        | [`stats.csv`](https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/region-analysis/states/stats.csv)   | [`stats.geojson`](https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/region-analysis/states/stats.geojson)   |
| Statistics across **CONUS**         | [`stats.csv`](https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/region-analysis/nation/stats.csv)   | [`stats.geojson`](https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/vector/production/v0.13.1/region-analysis/nation/stats.geojson)   |

### Schema

| Variable                            | Type    | Description                                                                                                                                                                                                                        |
| ----------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Region identifiers**              |         | _Identifying information for region--a county, census tract, or census block--represented in each row_                                                                                                                             |
| `GEOID`                             | int     | [Geographic Identifier](https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html) for region                                                                                                                |
| `building_count`                    | int     | Total number of buildings in region                                                                                                                                                                                                |
| `centroid_longitude`                | degrees | Longitude of centroid of region                                                                                                                                                                                                    |
| `centroid_latitude`                 | degrees | Latitude of centroid of region                                                                                                                                                                                                     |
| **Risk values**                     |         | _Risk values from output, input, or comparison datasets_                                                                                                                                                                           |
| `avg_wind_risk_2011`                | float   | Mean risk score (wind-informed, 2011 climate conditions) across all buildings in region                                                                                                                                            |
| `avg_wind_risk_2047`                | float   | Mean risk score (wind-informed, 2047 climate conditions) across all buildings in region                                                                                                                                            |
| `avg_burn_probability_2011`         | float   | Mean burn probability (wind-informed, 2011 climate conditions) across all buildings in region                                                                                                                                      |
| `avg_burn_probability_2047`         | float   | Mean burn probability (wind-informed, 2047 climate conditions) across all buildings in region                                                                                                                                      |
| `avg_conditional_risk_usfs`         | float   | Mean USFS Conditional Risk to Potential Structures (cRPS) from Scott (2024)                                                                                                                                                        |
| `avg_burn_probability_usfs_2011`    | float   | Mean burn probability (2011 climate conditions) across all buildings in region from Riley et al. (2025)                                                                                                                            |
| `avg_burn_probability_usfs_2047`    | float   | Mean burn probability (2047 climate conditions) across all buildings in region from Riley et al. (2025)                                                                                                                            |
| `median_wind_risk_2011`             | float   | Median risk score (wind-informed, 2011 climate conditions) across all buildings in region                                                                                                                                          |
| `median_wind_risk_2047`             | float   | Median risk score (wind-informed, 2047 climate conditions) across all buildings in region                                                                                                                                          |
| `median_burn_probability_2011`      | float   | Median burn probability (wind-informed, 2011 climate conditions) across all buildings in region                                                                                                                                    |
| `median_burn_probability_2047`      | float   | Median burn probability (wind-informed, 2047 climate conditions) across all buildings in region                                                                                                                                    |
| `median_conditional_risk_usfs`      | float   | Median USFS Conditional Risk to Potential Structures (cRPS) from Scott (2024)                                                                                                                                                      |
| `median_burn_probability_usfs_2011` | float   | Median burn probability (2011 climate conditions) across all buildings in region from Riley et al. (2025)                                                                                                                          |
| `median_burn_probability_usfs_2047` | float   | Median burn probability (2047 climate conditions) across all buildings in region from Riley et al. (2025)                                                                                                                          |
| **Histogram values**                |         | _Index in array corresponds to integer risk score (e.g., first value reflects number buildings with `0` score, next value reflects number buildings with `1` score, final value reflects number buildings with `10` score, etc.)._ |
| `wind_risk_2011_hist`               | float[] | Count of buildings with each risk score (wind-informed, 2011 climate conditions).                                                                                                                                                  |
| `wind_risk_2047_hist`               | float[] | Median risk score (wind-informed, 2047 climate conditions) across all buildings in                                                                                                                                                 |
| `burn_probability_2011_hist`        | float[] | Median risk score (wind-informed, 2047 climate conditions) across all buildings in                                                                                                                                                 |
| `burn_probability_2047_hist`        | float[] | Median risk score (wind-informed, 2047 climate conditions) across all buildings in                                                                                                                                                 |
| `conditional_risk_usfs_hist`        | float[] | Median risk score (wind-informed, 2047 climate conditions) across all buildings in                                                                                                                                                 |
| `burn_probability_usfs_2011_hist`   | float[] | Median risk score (wind-informed, 2047 climate conditions) across all buildings in                                                                                                                                                 |
| `burn_probability_usfs_2047_hist`   | float[] | Median risk score (wind-informed, 2047 climate conditions) across all buildings in                                                                                                                                                 |

## Downloading subsetted data in the web tool

The [web tool](https://ocr.carbonplan.org/) can be used to access region-specific, subsetted downloads.

### Steps

1. Using the map or search bar, navigate to region of interest.
2. Scroll to the `Risk in the region` section in the sidebar.
3. Select your region of interest (county, census tract, or census block) and view trends inline.
4. Click `CSV ↓` or `GeoPackage ↓` to download building-level data for the selected region.

![](../assets/web-data-downloads.png)

### Schema

| Variable                     | Type    | Description                                                                                                                                                                        |
| ---------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Building identifiers**     |         |                                                                                                                                                                                    |
| `GEOID`                      | int     | [Geographic Identifier](https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html) representing building (multiple buildings might belong to the same GEOID) |
| `centroid_longitude`         | degrees | Longitude of building centroid                                                                                                                                                     |
| `centroid_latitude`          | degrees | Latitude of building centroid                                                                                                                                                      |
| **Risk values**              |         |                                                                                                                                                                                    |
| `wind_risk_2011`             | float   | Risk score (wind-informed, 2011 climate conditions)                                                                                                                                |
| `wind_risk_2047`             | float   | Risk score (wind-informed, 2047 climate conditions)                                                                                                                                |
| `burn_probability_2011`      | float   | Burn probability (wind-informed, 2011 climate conditions)                                                                                                                          |
| `burn_probability_2047`      | float   | Burn probability (wind-informed, 2047 climate conditions)                                                                                                                          |
| `conditional_risk_usfs`      | float   | USFS Conditional Risk to Potential Structures (cRPS) from Scott (2024)                                                                                                             |
| `burn_probability_usfs_2011` | float   | Burn probability (2011 climate conditions) from Riley et al. (2025)                                                                                                                |
| `burn_probability_usfs_2047` | float   | Burn probability (2047 climate conditions) from Riley et al. (2025)                                                                                                                |
