# Fire Risk Methods Overview

This page explains the high-level methodology used to compute building-level wildfire risk scores for CONUS. It provides a conceptual overview with references to more detailed documentation pages.

## Summary

OCR's fire risk model computes building-level wildfire risk scores by:

1. Taking baseline burn probability data from USFS and spreading it into developed areas using wind-informed blurring
2. Multiplying the wind-adjusted burn probability by USFS conditional risk values (cRPS)
3. Sampling the resulting risk surface at building locations

The model produces risk scores (RPS: Risk to Potential Structures) representing the expected net value change for a hypothetical structure at each location. The scores account for directional fire spread patterns driven by prevailing winds.

## Conceptual Framework

### Risk to Potential Structures (RPS)

The model calculates **Risk to Potential Structures (RPS)**, defined as:

$$
\text{RPS} = \text{BP} \times \text{cRPS}
$$

Where:

- **BP (Burn Probability)**: Annual likelihood that a given pixel burns, derived from USFS wildfire simulations
- **cRPS (Conditional Risk to Potential Structures)**: The conditional net value change for a hypothetical structure if it were to burn, from USFS data

RPS represents the **expected net value change** per year for a generic structure at each location. It combines both probability (how likely fire is) and consequence (how much damage would occur).

!!! note "Key limitation"

    This approach models risk to a hypothetical "potential structure" rather than actual buildings with specific characteristics. Building-level attributes (materials, retrofits, defensible space) are not included.

### Wind-Adjusted Fire Spread

A key innovation is incorporating **wind-driven fire spread patterns** into the burn probability:

1. **Fire weather analysis**: Identify wind directions during high fire weather conditions (99th percentile FFWI) from CONUS404 data
2. **Elliptical spread kernels**: Apply oval-shaped blurring filters (inspired by Richards 1990) pointing in eight cardinal/ordinal directions
3. **Upwind spreading**: For each pixel, calculate zonal mean BP from upwind areas weighted by fire-weather wind direction frequencies
4. **Iterate spreading**: Repeat blurring 3 times to spread BP up to ~1.5 km into non-burnable (developed) areas

This differs from the uniform circular blurring in USFS methods. The wind-informed approach better represents how embers transport fire downwind from wildland into developed areas.

## Input Data

### Raster Datasets

- **USFS Burn Probability (BP)**: 270m resolution, present-day (circa 2011) and future (circa 2047) from Riley et al. (2025)
    - Only available for wildland areas; non-burnable areas have BP = 0
- **USFS Conditional Risk to Potential Structures (cRPS)**: 30m resolution, present-day (circa 2023) from Scott et al. (2024)
    - Represents conditional net value change if a hypothetical structure at that pixel were to burn
    - Available for all of CONUS

- **CONUS404 Meteorological Data**: 4km gridded hourly data (1979-2022) from Rasmussen et al. (2023)
    - U and V wind components, temperature, relative humidity
    - Used to calculate Fosberg Fire Weather Index (FFWI)

### Vector Datasets

- **Building Footprints**: Overture Maps
    - Coverage: CONUS-wide
    - Attributes: Building geometries, centroid coordinates for risk sampling

See [Data Sources and Provenance](./data-sources-and-provenance.md) for detailed information on data access, versions, and preprocessing.

## Processing Workflow

The calculation follows these steps:

### 1. Fire Weather Wind Analysis

- Calculate Fosberg Fire Weather Index (FFWI) for every hour 1979-2022
- For each 4km pixel, identify 99th percentile FFWI threshold
- Extract wind directions for all hours exceeding that threshold ("fire-weather winds")
- Bin fire-weather winds into 8 cardinal/ordinal directions
- Create distribution of fire-weather wind directions for each pixel

### 2. Upscale and Prepare BP

- Convert 270m USFS BP raster to 30m resolution
- Identify "non-burnable" pixels (where BP = 0 in Riley et al. data)

### 3. Wind-Informed BP Spreading

For each 30m pixel:

- Extract nearest-neighbor 4km fire-weather wind distribution
- Create 8 oval-shaped blurring filters (elliptical wavelets) pointing in 8 directions
    - Each filter represents wind coming FROM that direction (spreading BP downwind TO the pixel)
    - Distance from pixel to far side of oval along major axis: 510m
- For each direction, apply upwind filter to calculate zonal mean BP
- Weight the 8 smeared BP values by fire-weather wind direction frequencies
- **Repeat this process 3 times** → maximum spread of ~1.5 km into non-burnable areas

### 4. Calculate RPS

- Multiply wind-adjusted BP by 30m cRPS raster
- Result: 30m RPS (Risk to Potential Structures) for present and future
- RPS = expected net value change per year for a hypothetical structure

### 5. Sample at Building Locations

- Intersect 30m RPS raster with Overture Maps building footprints
- Assign risk score to each structure based on value at building centroid

### 6. Convert to Categorical Scores

- Convert continuous RPS values to categorical risk scores (1-10 scale)
- Scores based on percentile bins of RPS across full CONUS domain

For detailed implementation steps, see the [Implementation](./implementation.ipynb) page.

## Spatial Processing Architecture

The model uses a spatial chunking system for efficient parallel processing:

- CONUS is divided into 595 processing regions (30m resolution chunks)
- Each region is processed independently using distributed compute (Coiled/Dask)
- Outputs are stored in Icechunk (for rasters) and GeoParquet (for vectors)
- Failed regions can be reprocessed without affecting completed work

See [Horizontal Scaling via Spatial Chunking](./horizontal-scaling-via-spatial-chunking.ipynb) for details on the parallelization strategy.

## Outputs

### Building-Level Outputs

- **GeoParquet files** containing:
    - Building geometries and IDs
    - Wind-adjusted fire risk scores
    - Baseline USFS risk metrics for comparison
    - Metadata (region ID, processing timestamp)

### Regional Aggregations

- **PMTiles** for interactive web visualization:
    - 1, 15, and 30-year time horizons
    - County and census tract aggregations
    - Pre-rendered at multiple zoom levels

- **Summary statistics** (GeoParquet, GeoJSON, CSV):
    - Buildings at risk counts by jurisdiction
    - Percentile distributions of risk scores
    - Comparison metrics (wind-adjusted vs baseline)

## Key Assumptions and Limitations

- **"Generic" structure assumption**: Risk scores represent a hypothetical structure, not actual building characteristics
- **Static building inventory**: Does not account for new construction or demolition post-data-collection
- **Historical wind climatology**: Uses past wind patterns (1979-2022) as proxy for fire weather
- **No explicit fire suppression**: Model does not account for firefighting efforts
- **Wildfire focus only**: Does not include structure-to-structure fire spread in WUI
- **99th percentile FFWI**: Fire weather threshold may not perfectly capture conditions during largest fires
- **Point-specific wind data**: Wind directions at pixel B determine spreading, regardless of wind at upwind pixel A (see limitations discussion)

For a detailed discussion of caveats, see [Caveats & Limitations](./limitations.md).

## Factors Not Included

The risk scores described above represent risk to a **"potential" or "generic" structure**. They do NOT account for factors that drive actual risk at specific buildings up or down:

| Factor                                                                                 | Risk Impact | Notes                                                                                            |
| -------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------ |
| Building retrofit (fire-resistant materials, ember-resistant vents)                    | Lower       | Could significantly reduce actual risk                                                           |
| Community emergency response capabilities                                              | Lower       | Firefighting effectiveness not modeled                                                           |
| Previous fire / fuel reduction                                                         | Lower       | Changes to burnable landscape not captured                                                       |
| Access limitations (road conditions, remote locations)                                 | Higher      | Evacuation and response challenges                                                               |
| Building-specific characteristics (materials, defensible space, vegetation management) | Variable    | Generic structure assumption means individual building attributes ignored                        |
| Development impacts on BP/cRPS                                                         | Variable    | Risk scores for undeveloped land assume sole structure; large developments alter fire conditions |

**Interpretation guidance**: The risk score at a given address should be understood as risk to a generic building at that location, leaving it to users to assess how their actual building compares to that generic baseline.

For validation approaches and uncertainty quantification, see [Validation and Uncertainty](./validation-and-uncertainty.ipynb).

## References and Further Reading

### Primary Data Sources

- Riley et al. (2025): USFS Burn Probability rasters. [RDS-2025-0006](https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0006)
- Scott et al. (2024): USFS Conditional Risk to Potential Structures (30m)
- Rasmussen et al. (2023, 2024): CONUS404 4km gridded meteorology
- Overture Maps: Building footprints ([overturemaps.org](https://overturemaps.org))

### Methodological Background

- Fosberg (1978): Fosberg Fire Weather Index
- Richards (1990): Elliptical fire spread wavelets
- Finney (2005): "The challenge of quantitative risk analysis for wildland fire"
- Scott and Thompson (2015): Expected net value change / RPS definition

### Related Documentation

- [Data Sources and Provenance](./data-sources-and-provenance.md): Detailed data access and preprocessing
- [Implementation](./implementation.ipynb): Step-by-step calculation details
- [Validation and Uncertainty](./validation-and-uncertainty.ipynb): Model evaluation and uncertainty analysis
- [Horizontal Scaling via Spatial Chunking](./horizontal-scaling-via-spatial-chunking.ipynb): Parallel processing architecture
- [Data Pipeline Tutorial](../../tutorials/data-pipeline.md): Runnable tutorial for reproducing calculations
