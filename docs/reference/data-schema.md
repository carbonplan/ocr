# Data schema

Open Climate Risk (OCR) produces two primary types of output data: **raster (tensor) datasets** and **vector (polygon) datasets**. This page documents the structure, variables, and schema of both output types.

## Overview

```mermaid
%%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#2563eb','primaryTextColor':'#1f2937','primaryBorderColor':'#3b82f6','lineColor':'#6b7280','secondaryColor':'#7c3aed','tertiaryColor':'#10b981','background':'#ffffff','mainBkg':'#f3f4f6','secondBkg':'#e5e7eb','tertiaryBkg':'#d1d5db','primaryTextColor':'#111827','lineColor':'#6b7280','textColor':'#374151','mainContrastColor':'#1f2937','darkMode':false}}}%%
graph TB
    %% Input Data Sources
    subgraph Inputs["<b>Input Data Sources</b>"]
        USFS[USFS Fire Risk<br/>Scott et al. 2024<br/>Riley et al. 2025]
        CONUS[CONUS404 Climate<br/>Rasmussen et al. 2023]
        Buildings[Overture Maps<br/>Building Footprints]
    end

    %% Processing Pipeline
    subgraph Pipeline["<b>OCR Processing Pipeline</b>"]
        ProcessRegion[Process Region<br/>Wind-Adjusted Risk Calculation]
        WindCalc[Wind Direction<br/>Distribution]
        Sample[Sample Risk Values<br/>at Building Locations]
    end

    %% Raster Outputs
    subgraph RasterOutputs["<b>Raster Datasets (30m resolution)</b>"]
        RiskLayers["<b>Fire Risk Variables</b><br/>• rps_2011, rps_2047<br/>• bp_2011, bp_2047<br/>• Reference: rps_scott, crps_scott<br/>• Reference: bp_2011_riley, bp_2047_riley"]
        WindDist["<b>Wind Distribution</b><br/>• wind_direction_distribution<br/>• 8 cardinal/ordinal directions<br/>• Derived from CONUS404"]
    end

    %% Vector Outputs
    subgraph VectorOutputs["<b>Vector Datasets (polygon)</b>"]
        BuildingRisk["<b>Building-Level Risk</b><br/>• Same variables as raster<br/>• Sampled at building centroids<br/>• CONUS-wide coverage<br/>"]
    end

    %% Storage Formats
    subgraph Storage["<b>Storage Formats</b>"]
        Icechunk[("<b>Icechunk</b><br/>Zarr-based<br/>S3-backed<br/>Versioned")]
        GeoParquet[("<b>GeoParquet</b><br/>Hive-partitioned")]
    end

    %% Data Flow
    USFS --> ProcessRegion
    CONUS --> ProcessRegion
    CONUS --> WindCalc
    Buildings --> Sample

    ProcessRegion --> RiskLayers
    WindCalc --> WindDist
    ProcessRegion --> Sample

    RiskLayers --> Icechunk
    WindDist --> Icechunk
    Sample --> BuildingRisk
    BuildingRisk --> GeoParquet

    %% Styling
    classDef input fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#075985
    classDef process fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#5b21b6
    classDef raster fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,color:#92400e
    classDef vector fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#166534
    classDef storage fill:#fce7f3,stroke:#ec4899,stroke-width:2px,color:#9f1239

    class USFS,CONUS,Buildings input
    class ProcessRegion,WindCalc,Sample process
    class RiskLayers,WindDist raster
    class BuildingRisk vector
    class Icechunk,GeoParquet storage
```

## Raster (tensor) datasets

Raster datasets are gridded geospatial layers stored at 30m resolution in EPSG:4326 (WGS84) projection. These datasets are organized by region and stored in the Icechunk format.

### Spatial characteristics

| Property           | Value                  |
| ------------------ | ---------------------- |
| **Resolution**     | 30m (~0.00028 degrees) |
| **Projection**     | EPSG:4326 (WGS84)      |
| **Extent**         | CONUS                  |
| **Chunking**       | Regional chunks        |
| **Storage Format** | Icechunk (Zarr-based)  |

### Fire risk variables

The primary output dataset contains the following variables. To support transparency we also include previously published datasests we used as inputs or for comparison. For clarity, we append a `variable_{name}` modifier to any variable name describing previously-published data. We are the authors of any data without a `_{name}` modifier.

#### Core risk variables

| Variable   | Type    | Units         | Description                                                                                                           |
| ---------- | ------- | ------------- | --------------------------------------------------------------------------------------------------------------------- |
| `rps_2011` | float32 | %             | Annual relative risk to potential structures (RPS) for ~2011 climate conditions. Calculated as `bp_2011 × crps_scott` |
| `rps_2047` | float32 | %             | Annual risk to potential structures (RPS) for ~2047 climate conditions. Calculated as `bp_2047 × crps_scott`          |
| `bp_2011`  | float32 | dimensionless | Annual burn probability for ~2011 climate conditions                                                                  |
| `bp_2047`  | float32 | dimensionless | Annual burn probability for ~2047 climate conditions                                                                  |

#### Reference variables (data from USFS and Wildfire Risk to Communities project)

| Variable        | Type    | Units         | Description                                                               |
| --------------- | ------- | ------------- | ------------------------------------------------------------------------- |
| `rps_scott`     | float32 | %             | Annual risk to potential structures from Scott et al., (2024)             |
| `crps_scott`    | float32 | %             | Conditional risk to potential structures (cRPS) from Scott et al., (2024) |
| `bp_2011_riley` | float32 | dimensionless | Burn probability for ~2011 from Riley et al. (2025) (RDS-2025-0006)       |
| `bp_2047_riley` | float32 | dimensionless | Burn probability for ~2047 from Riley et al. (2025)                       |

#### Coordinate variables

| Variable    | Type    | Description                          |
| ----------- | ------- | ------------------------------------ |
| `latitude`  | float64 | Latitude in decimal degrees (WGS84)  |
| `longitude` | float64 | Longitude in decimal degrees (WGS84) |

### Wind direction distribution dataset

A separate dataset provides the statistical distribution of wind directions during fire-weather conditions:

| Variable                      | Type    | Dimensions                            | Description                                                                                                            |
| ----------------------------- | ------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `wind_direction_distribution` | float32 | (latitude, longitude, wind_direction) | Fraction of fire-weather hours coming from each of 8 cardinal/ordinal directions derived from Rasmussen et al., (2023) |

**Wind direction dimension:**
The `wind_direction` coordinate contains 8 direction labels: `['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']`

**Properties:**

- Values sum to 1.0 for all pixels (normalized probability distribution)
- Derived from CONUS404 data (Rasmussen et al, 2023) using 99th percentile Fosberg Fire Weather Index (FFWI) as threshold

### Data processing flow

```mermaid
%%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#2563eb','primaryTextColor':'#1f2937','primaryBorderColor':'#3b82f6','lineColor':'#6b7280','secondaryColor':'#7c3aed','tertiaryColor':'#10b981','background':'#ffffff','mainBkg':'#f3f4f6','secondBkg':'#e5e7eb','tertiaryBkg':'#d1d5db','primaryTextColor':'#111827','lineColor':'#6b7280','textColor':'#374151','mainContrastColor':'#1f2937','darkMode':false}}}%%
flowchart LR
    %% Input Data
    subgraph Inputs["<b>Input Data</b>"]
        BP_Riley["<b>Riley et al. (2025)</b><br/>Burn Probability<br/>• bp_2011_riley<br/>• bp_2047_riley"]
        CRPS["<b>Scott et al. (2024)</b><br/>Conditional RPS<br/>• crps_scott"]
        WindDist["<b>CONUS404</b><br/>Wind Distribution<br/>• 8 directions<br/>• Fire-weather hours"]
    end

    %% Processing Steps
    subgraph Processing["<b>Wind-Adjustment Processing</b>"]
        Convolve["<b>Directional Convolution</b><br/>Apply elliptical kernel<br/>for each wind direction<br/>30m resolution"]
        Weight["<b>Weighted Composite</b><br/>Combine 8 directions<br/>using wind frequency<br/>as weights"]
    end

    %% Intermediate Results
    WindAdjBP["<b>Wind-Adjusted BP</b><br/>bp_2011, bp_2047<br/>Accounts for directional<br/>fire spread"]

    %% Final Output
    FinalRPS["<b>Risk to Potential Structures</b><br/>rps_2011, rps_2047<br/>Annual risk [%]<br/>Ready for sampling"]

    %% Data Flow
    BP_Riley --> Convolve
    WindDist --> Weight
    Convolve --> Weight
    Weight --> WindAdjBP
    WindAdjBP --> |"Multiply"| FinalRPS
    CRPS --> |"Multiply"| FinalRPS

    %% Styling
    classDef input fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#075985
    classDef process fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#5b21b6
    classDef intermediate fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,color:#92400e
    classDef output fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#166534

    class BP_Riley,CRPS,WindDist input
    class Convolve,Weight process
    class WindAdjBP intermediate
    class FinalRPS output
```

## Vector (polygon) datasets

Vector datasets contain building-level risk samples stored as a consolidated GeoParquet file covering all buildings across CONUS.

### Schema

#### Geometry column

| Column     | Type          | Description                            |
| ---------- | ------------- | -------------------------------------- |
| `geometry` | WKB (Polygon) | Building polygon location in EPSG:4326 |

#### Risk attribute columns

Vector datasets contain the same risk variables as raster datasets, sampled at each building location:

| Column          | Type    | Description                                                                        |
| --------------- | ------- | ---------------------------------------------------------------------------------- |
| `rps_2011`      | float32 | Annual risk to potential structures for ~2011 at building location                 |
| `rps_2047`      | float32 | Annual risk to potential structures for ~2047 at building location                 |
| `bp_2011`       | float32 | Annual burn probability for ~2011 at building location                             |
| `bp_2047`       | float32 | Annual burn probability for ~2047 at building location                             |
| `rps_scott`     | float32 | Annual risk to potential structures (Scott et al., 2024) at building location      |
| `crps_scott`    | float32 | Conditional risk to potential structures (Scott et al., 2024) at building location |
| `bp_2011_riley` | float32 | Annual burn probability ~2011 (Riley et al, 2025) at building location             |
| `bp_2047_riley` | float32 | Annual burn probability ~2047 (Riley et al, 2025) at building location             |

#### Region column

| Column  | Type   | Description                                                                                  |
| ------- | ------ | -------------------------------------------------------------------------------------------- |
| `GEOID` | string | Census TIGER block holding the building's sampling point; buildings in no CONUS block are dropped |

#### Map columns

The file also carries the columns the web map draws tiles from. `lod` is about 28% of the file,
and `SELECT *` or a read without a column list downloads it, so leave these columns out of analysis
reads: `SELECT * EXCLUDE (bbox, hilbert, lod)` in DuckDB, or `columns=[...]` in GeoPandas. A query can
still filter on `bbox` without selecting it.

| Column    | Type   | Description                                                                                                                            |
| --------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `bbox`    | struct | Bounding box covering (float32, rounded outward), with page statistics                                                                 |
| `hilbert` | int64  | Hilbert index of each building's tile, the file's sort order                                                                           |
| `lod`     | struct | Simplified geometry for each overview zoom, with `rps_2011` and `rps_2047` at half precision, in the [geoparquet-lod][lod] packed form |

[lod]: https://github.com/Shane98c/geoparquet-lod

### Storage characteristics

| Property              | Value                                                                          |
| --------------------- | ------------------------------------------------------------------------------ |
| **Format**            | GeoParquet (schema version 1.1.0)                                              |
| **Compression**       | zstd                                                                           |
| **Geometry Encoding** | WKB                                                                            |
| **Spatial Index**     | Covering bounding box (bbox), Hilbert order and a geoparquet-lod spatial index |
| **Coverage**          | CONUS-wide, single consolidated file                                           |
| **Aggregation**       | Consolidated from regional processing via DuckDB and geoparquet-lod            |

### Data quality

- Buildings with NaN values (outside CONUS) are excluded
- Building locations sourced from Overture Maps dataset

### File location

The consolidated building dataset is available at:

```text
output/{risk}/v{version}/vector/buildings.geoparquet
```

This single-file format enables:

- Efficient CONUS-wide spatial queries
- Direct access for analysis tools and workflows
- Simplified data distribution and versioning

## Regional statistics datasets

Regional statistics are published as GeoParquet, one dataset per region level, with each region's full
TIGER geometry. Each region gets `building_count`, plus `{var}_mean`, `{var}_median` and `{var}_missing`
(NULL or NaN count) for every building value and `{var}_hist` for the risk's histogram variables.
Buildings roll up from their block `GEOID`, so levels nest. The web map reads its region statistics from these files.

### Layout

```text
output/{risk}/v{version}/vector/regions/
  nation.parquet
  states.parquet
  counties/state_fips=SS/data.parquet
  tracts/state_fips=SS/county_fips=CCC/data.parquet
  blocks/state_fips=SS/county_fips=CCC/data.parquet
```

Partitions are Hive-style, so a query over a level's glob can filter on `state_fips` and `county_fips` and read
only the matching files:

```sql
SELECT GEOID, building_count, rps_2011_median
FROM read_parquet('output/fire/v{version}/vector/regions/blocks/*/*/data.parquet', hive_partitioning = true)
WHERE state_fips = '08' AND county_fips = '013';
```

### Columns

| Column                                         | Type                          | Description                                      |
| ---------------------------------------------- | ----------------------------- | ------------------------------------------------ |
| `GEOID`                                        | string                        | Census GEOID, with leading zeros                 |
| `STUSPS`                                       | string                        | State abbreviation (states only)                 |
| `NAME`                                         | string                        | Region name (nation, states and counties)        |
| `{var}_hist` (e.g. `rps_2011_hist`)            | list&lt;int64&gt;             | Building count per bin: see below                |
| `bbox`                                         | struct                        | Bounding box covering, with page statistics      |
| `geometry`                                     | WKB (Polygon or MultiPolygon) | Full region geometry in EPSG:4269 (NAD83, from TIGER) longitude/latitude |

There are no `longitude` and `latitude` columns; the centroid follows from the geometry.

A hist with bin edges `b` (file metadata `bins`) is `[x = 0, x <= b[0], b[0] < x <= b[1], ..., x > b[-2]]`:
negatives fall in the first nonzero bin and values above `b[-2]` in the last, so it sums to the
non-null count. The last edge `b[-1]` is only a display cap for the last bin's label: values above it
are still counted there (fire's legacy hist dropped them). NaN counts as missing. File metadata also carries `version`, `tiger_vintage` and the
Overture release.

### Storage characteristics

| Property        | Value                                                                                  |
| --------------- | -------------------------------------------------------------------------------------- |
| **Format**      | GeoParquet (schema version 1.1.0)                                                      |
| **Compression** | zstd                                                                                   |
| **Row order**   | GEOID                                                                                  |
| **Indexes**     | Statistics and page indexes on `GEOID` and `bbox`                                      |
| **Pages**       | At most 1 row per page for nation, states and counties, 8 for tracts and 32 for blocks |

## Data validation

### Expected value ranges

| Variable                                        | Expected Range | Notes                                                                                            |
| ----------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------ |
| Risk to potential structures (RPS)              | [0, 100]       | Annual risk of loss [%] to potential structures. Product of BP and cRPS.                         |
| Conditional risk to potential structures (cRPS) | [0, 100]       | Risk of loss [%] to a hypothetical structure if it were to burn                                  |
| Burn probability (BP)                           | [0, 1]         | Annual likelihood [-] of a pixel burning                                                         |
| Wind Distribution                               | [0, 1]         | Sums to 1.0 per pixel (normalized probability distribution across 8 cardinal/ordinal directions) |

### Quality checks

1. **Spatial consistency**: All raster layers share identical coordinate systems and extents
2. **Missing data**: NaN values appear only in unburnable areas (water, urban, etc.)

3. **Normalization**: Wind direction distributions sum to 1.0 (within tolerance of 1e-5) where valid

## Metadata attributes

All datasets include descriptive metadata attributes:

- `description`: Human-readable description of the variable
- `long_name`: Extended variable name
- `units`: Physical units (if applicable)
- `composition`: Method used for compositing (e.g., "weighted")
- `direction_labels`: Cardinal/ordinal direction labels for wind data
- `weights_source`: Source of weights used in calculations

## Access patterns

### Raster data

- **By region**: Query specific regional chunks using latitude/longitude slices
- **Full CONUS**: Access complete dataset via Icechunk storage

### Vector data

- **Full dataset**: Query the consolidated CONUS-wide building dataset
- **Spatial query**: Use bounding box attributes for efficient spatial filtering; the file is in Hilbert order, so a bounding box filter skips most of it
- **Attribute query**: Filter by risk threshold using Parquet predicate pushdown with DuckDB or similar tools
- **Regional subset**: Extract specific areas using spatial predicates on latitude/longitude

## Related documentation

- [Access data](../access-data.md): Information on accessing and downloading datasets
- [Deployment](deployment.md): Details on data storage infrastructure
