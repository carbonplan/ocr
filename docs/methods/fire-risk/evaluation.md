# Evaluation

This section contains comprehensive evaluation and benchmarking analyses of our fire risk estimates. These notebooks compare our results against historical fire data, other datasets, and provide detailed statistical assessments.

## Overview

Our evaluation approach includes multiple independent analyses to validate the quality and reliability of our fire risk estimates:

1. **Comparison with historical fire data** - Benchmarking against 70+ years of actual burn perimeters
2. **Cross-dataset validation** - Comparing with established fire risk datasets
3. **Regional deep-dives** - Detailed analysis of specific geographic areas and historical events
4. **Methodological validation** - Statistical assessment of our scoring and classification approaches

## Evaluation Notebooks

### [Benchmarking](benchmarking.ipynb)

Comprehensive comparison of our burn probability estimates against historical U.S. fire perimeter data. This analysis adapts methods from [Moran et al. 2025](https://www.nature.com/articles/s41598-025-07968-6) to benchmark our model-derived burn probabilities, examining both all pixels and specifically "non-burnable" areas where we extended estimates beyond the original Riley et al. (2025) coverage.

**Key analyses:**

-   Distribution of burn probability in historically burned vs. unburned areas
-   Performance assessment in areas we designated as burnable
-   Statistical comparison with 70+ year fire history

### [California Comparison](california-comparison.ipynb)

Detailed comparison of our risk estimates with two authoritative California datasets: the Wildfire Risk to Communities (WRC) project and [California Fire Hazard Severity Zones](https://osfm.fire.ca.gov/what-we-do/community-wildfire-preparedness-and-mitigation/fire-hazard-severity-zones) from CAL FIRE.

**Key analyses:**

-   Census tract-level concordance analysis using Kendall's Tau
-   Spatial patterns of agreement and disagreement
-   Regional variation in performance metrics

**Key features:**

-   Monotonically descending bin prevalence with increasing risk scores
-   Distribution-based bin design using building-level data
-   Comparison with other scoring approaches

### [Comparing Risk Rasters](compare-risk-rasters.ipynb)

Visual comparison of our 30m resolution risk rasters with those from the Wildfire Risk to Communities project. This notebook showcases regions where the datasets differ and explains the underlying causes, including effects of wind modeling and development patterns.

**Key analyses:**

-   Areas of low correlation between datasets
-   Regions with high and low bias
-   Historical fire locations (Eaton Fire, Marshall Fire, Camp Fire)
-   "Wind effect" vs. "development effect" attribution

### [Comparing Risk at Buildings](compare-risk-buildings.ipynb)

Building-level comparison of risk estimates, examining how our approach differs from other datasets when evaluated at the scale of individual structures.

## Key Findings

Our evaluation demonstrates that:

1. **Historical concordance**: Areas with higher burn probability in our estimates correspond well with areas that have historically burned, validating the spatial distribution of risk
2. **Regional performance**: Our estimates show strong concordance with established California fire hazard maps, with performance comparable to or exceeding other national datasets
3. **Wind modeling impact**: Our incorporation of wind effects produces meaningful differences in developed areas, particularly in regions prone to wind-driven fires
4. **Scoring validity**: Our categorical scoring system effectively captures the full range of risk while maintaining interpretability across different risk levels

## References

-   Finney, M.A., et al. (2011). A simulation of probabilistic wildfire risk components for the continental United States. Stochastic Environmental Research and Risk Assessment.
-   Moran, C.J., et al. (2025). Benchmarking burn probability maps in California using historical fire perimeters. Scientific Reports.
-   Riley, K.L., et al. (2025). Wildfire Risk to Communities methodology and data products.

```{toctree}
:hidden:
:maxdepth: 1

benchmarking
benchmarking-process-inputs
benchmarking-make-inputs
california-comparison
compare-risk-rasters
compare-risk-buildings
```
