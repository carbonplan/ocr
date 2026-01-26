# Getting Started

This guide helps you get started with accessing and using Open Climate Risk fire risk data for the continental United States.

## What is Open Climate Risk?

**Open Climate Risk** is CarbonPlan's platform for analyzing building-level wildfire risk across CONUS. It includes:

- **Building-level fire risk** for ~160 million structures
- **Wind-adjusted fire spread modeling** that accounts for directional fire propagation
- **Multiple output formats**: Interactive web maps, downloadable datasets, and cloud-native data access
- **Present and future scenarios**: Current conditions (circa 2011) and future projections (circa 2047)

## Quick Access Options

### Option 1: Explore the Web Tool

The fastest way to explore Open Climate Risk data is through our [interactive web map](https://carbonplan.org/research/climate-risk). The web tool allows you to:

- Search for specific addresses or locations
- View building-level risk scores on a 0-10 scale
- Explore state, county, census tract, and census block aggregations

### Option 2: Access Production Data

If you want to analyze Open Climate Risk data programmatically, you can access our production datasets directly from cloud storage using Python.

## Accessing Production Data

Open Climate Risk output data is stored in [Icechunk](https://icechunk.io/), a versioned, cloud-native data format that works seamlessly with `Xarray` and `Zarr`.

### Prerequisites

You'll need Python with a few packages installed:

```bash
python -m pip install xarray icechunk
```

### Load the Dataset

Here's a minimal example to load Open Climate Risk wind-adjusted fire risk data:

```python
import icechunk
import xarray as xr

# Connect to production Icechunk repository
version = 'v0.13.1'  # Check GitHub releases for latest version
storage = icechunk.s3_storage(
    bucket='us-west-2.opendata.source.coop',
    prefix=f'carbonplan/carbonplan-ocr/output/fire-risk/tensor/production/{version}/ocr.icechunk',
    anonymous=True,
)

repo = icechunk.Repository.open(storage)
session = repo.readonly_session('main')

# Open the dataset
ds = xr.open_dataset(session.store, engine='zarr', chunks={})
ds
```

This gives you access to:

- **Raster datasets**: 30m resolution risk surfaces
- **Risk scores (RPS)**: Risk to Potential Structures values
- **Spatial coverage**: Full CONUS extent
- **Multiple variables**: Burn probability, conditional risk, wind-adjusted metrics

### Understanding the Data

The dataset contains several key variables:

- **`rps`**: Risk to Potential Structures (expected net value change per year)
- **`bp`**: Burn Probability (annual likelihood of burning)
- **`crps`**: Conditional Risk to Potential Structures (damage if fire occurs)
- Risk scores are for a "generic" or "potential" structure at each location

!!! note "Important Limitation"

    Risk scores represent a hypothetical structure and do NOT account for building-specific factors like construction materials, retrofits, or defensible space management.

## Next Steps

### For Data Users

- **[Working with Data](work-with-data.ipynb)**: Detailed guide on loading and analyzing Open Climate Risk datasets
- **[Data Schema](../reference/data-schema.md)**: Complete reference of available variables and metadata
- **[Data Downloads](../access-data.md)**: Direct download links and bulk access options

### For Researchers & Analysts

- **[Fire Risk Methods Overview](../methods/fire-risk/overview.md)**: Understand how risk scores are calculated
- **[Data Sources](../reference/data-sources.md)**: Learn about data sources

### For Developers

- **[Installation](installation.md)**: Set up project for local development
- **[Project Structure](../reference/project-structure.md)**: Understand the codebase
- **[Data Pipeline Tutorial](../tutorials/data-pipeline.md)**: Run the processing pipeline
- **[Working With Input Datasets](../how-to/work-with-input-datasets.md)**: View technical reference for working with input datasets

## Support

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/carbonplan/ocr/issues)
- **Questions & Discussions**: [GitHub Discussions](https://github.com/carbonplan/ocr/discussions)
- **General Inquiries**: [hello@carbonplan.org](mailto:hello@carbonplan.org)

## Available Data Versions

Check our [GitHub Releases](https://github.com/carbonplan/ocr/releases) page for:

- Latest data version numbers
- Release notes and changelogs
- Known issues and fixes
- Data format changes

---

_Ready to dive deeper? Check out the [Working with Data](work-with-data.ipynb) notebook for hands-on examples._
