# Fire risk methods overview

This page explains the high-level methodology used to compute building-level wildfire risk scores for CONUS. It provides a conceptual overview with references to more detailed documentation pages. For an full description of our methods, read the [methods document](https://carbonplan.org/research/climate-risk-fire-methods) that accompanied our [explainer article](https://carbonplan.org/research/climate-risk-explainer).

## Summary

The Open Climate Risk fire risk model computes building-level wildfire risk scores by:

1. Taking baseline burn probability data from [Riley et al. 2025](../../reference/data-sources.md#riley-et-al-2025) and spreading it into developed areas using wind-informed blurring
2. Multiplying the wind-adjusted burn probability by [Scott et al. 2024](../../reference/data-sources.md#scott-et-al-2024) conditional risk values (cRPS)
3. Sampling the resulting risk surface at building locations from the [Overture Maps Foundation buildings dataset](../../reference/data-sources.md#overture-maps-foundation-buildings-dataset)

The model produces risk scores (RPS: Risk to Potential Structures) representing the expected net value change for a hypothetical structure at each location. The scores account for directional fire spread patterns driven by prevailing winds.

## Conceptual Framework

### Risk to Potential Structures (RPS)

The model calculates **Risk to Potential Structures (RPS)**, defined as:

$$
\text{RPS} = \text{BP} \times \text{cRPS}
$$

Where:

-   **BP (Burn Probability)**: Annual likelihood that a given pixel burns, derived from wildfire simulations in [Riley et al. 2025](../../reference/data-sources.md#riley-et-al-2025)
-   **cRPS (Conditional Risk to Potential Structures)**: The conditional net value change for a hypothetical structure if it were to burn, from [Scott et al. 2024](../../reference/data-sources.md#scott-et-al-2024)

RPS represents the **expected net value change** per year for a generic structure at each location. It combines both probability (how likely fire is) and consequence (how much damage would occur).

:::{admonition} **Key limitation**
:class: note

This approach models risk to a hypothetical "potential structure" rather than actual buildings with specific characteristics. Building-level attributes (materials, retrofits, defensible space) are not included.
:::

### Wind-Adjusted Fire Spread

A key innovation is incorporating **wind-driven fire spread patterns** into the burn probability:

1. **Fire weather analysis**: Identify wind directions during high fire weather conditions (99th percentile FFWI) from [Rasmussen et al. 2023](../../reference/data-sources.md#rasmussen-et-al-2023)
2. **Elliptical spread kernels**: Apply oval-shaped blurring filters, inspired by Richards (1990), pointing in eight cardinal/ordinal directions
3. **Upwind spreading**: For each pixel, calculate zonal mean BP from upwind areas weighted by fire-weather wind direction frequencies
4. **Iterate spreading**: Repeat blurring 3 times to spread BP up to ~1.5 km into non-burnable (developed) areas

This differs from the uniform circular blurring in [Scott et al. 2024](../../reference/data-sources.md#scott-et-al-2024). The wind-informed approach better represents how embers transport fire downwind from wildland into developed areas.

## Processing Workflow

### 1. Fire Weather Wind Analysis

-   Calculate Fosberg Fire Weather Index (FFWI) for every hour 1979-2022
-   For each 4km pixel, identify 99th percentile FFWI threshold
-   Extract wind directions for all hours exceeding that threshold ("fire-weather winds")
-   Bin fire-weather winds into 8 cardinal/ordinal directions
-   Create distribution of fire-weather wind directions for each pixel

### 2. Upscale and Prepare BP

-   Convert 270m BP raster from [Riley et al. 2025](../../reference/data-sources.md#riley-et-al-2025) to 30m resolution
-   Identify "non-burnable" pixels (where BP = 0 in Riley et al. data)

### 3. Wind-Informed BP Spreading

For each 30m pixel:

-   Extract nearest-neighbor 4km fire-weather wind distribution
-   Create 8 oval-shaped blurring filters (elliptical wavelets) pointing in 8 directions
    -   Each filter represents wind coming FROM that direction (spreading BP downwind TO the pixel)
    -   Distance from pixel to far side of oval along major axis: 510m
-   For each direction, apply upwind filter to calculate zonal mean BP
-   Weight the 8 smeared BP values by fire-weather wind direction frequencies
-   **Repeat this process 3 times** → maximum spread of ~1.5 km into non-burnable areas

### 4. Calculate RPS

-   Multiply wind-adjusted BP by 30m cRPS raster
-   Result: 30m RPS (Risk to Potential Structures) for present and future
-   RPS = expected net value change per year for a hypothetical structure

### 5. Sample at Building Locations

-   Intersect 30m RPS raster with [Overture Maps building footprints](../../reference/data-sources.md#overture-maps-foundation-buildings-dataset)
-   Assign RPS to each structure based on value at building centroid

### 6. Convert to Categorical Scores

-   Convert continuous RPS values to categorical risk scores (0-10 scale)
-   Scores are calculated using percentile-based RPS bins defined [here](./score-bins.ipynb)

| Score | Criteria              |
| ----- | --------------------- |
| 0     | `RPS == 0`            |
| 1     | `0 < RPS < 0.01`      |
| 2     | `0.01 <= RPS < 0.02`  |
| 3     | `0.02 <= RPS < 0.035` |
| 4     | `0.035 <= RPS < 0.06` |
| 5     | `0.06 <= RPS < 0.1`   |
| 6     | `0.1 <= RPS < 0.2`    |
| 7     | `0.2 <= RPS < 0.5`    |
| 8     | `0.5 <= RPS < 1`      |
| 9     | `1 <= RPS < 3`        |
| 10    | `3 <= RPS <= 100`     |

## Spatial Processing Architecture

The model uses a spatial chunking system for efficient parallel processing:

-   CONUS is divided into 595 processing regions (30m resolution chunks)
-   Each region is processed independently using distributed compute (Coiled/Dask)
-   Outputs are stored in Icechunk (for rasters) and GeoParquet (for vectors)
-   Failed regions can be reprocessed without affecting completed work

See [Horizontal Scaling via Spatial Chunking](./horizontal-scaling-via-spatial-chunking.ipynb) for details on the parallelization strategy.

## Outputs

The pipeline produces data outputs in several formats:

| Option                    | Description                                                                             | Formats               |
| ------------------------- | --------------------------------------------------------------------------------------- | --------------------- |
| **Raster (tensor) data**  | Full gridded dataset spanning CONUS                                                     | Icechunk (Zarr-based) |
| **Vector (polygon) data** | Full buildings dataset spanning CONUS                                                   | GeoParquet            |
| **Regional statistics**   | Summary statistics for regions (state, county, census tract, census block) within CONUS | CSV, GeoJSON          |

For more information about each data output type, see the [Access Data](../../access-data.md) page.
