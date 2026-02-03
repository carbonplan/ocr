# Open Climate Risk

**Open Climate Risk** is CarbonPlan's platform for analyzing and visualizing climate-related risks at a building-level resolution across the Continental United States (CONUS).

## Quick Links

-   [**Map Tool**](https://carbonplan.org/research/climate-risk)
-   [**Explainer Article**](https://carbonplan.org/research/climate-risk-explainer)
-   [**GitHub Repository**](https://github.com/carbonplan/ocr)
-   [**Dataset Releases**](https://github.com/carbonplan/ocr/releases)

## Getting Started

::::{tab-set}

:::{tab-item} Using Open Climate Risk Data

If you want to **access and analyze fire risk data**:

1. Visit [Access Data](access-data.md) for an overview of all available formats
2. Read the [Working With Data](how-to/work-with-data.ipynb) guide to learn how to load production datasets
3. Explore the [Data Schema](reference/data-schema.md) to understand available variables

:::

:::{tab-item} Understanding Methods

If you want to **understand how fire risk is calculated**:

1. Read the [Fire Risk Overview](methods/fire-risk/overview.md) for conceptual background
2. Learn about how we performed [Evaluation](methods/fire-risk/evaluation.md) of our fire risk estimates
3. Review our [Data Sources](reference/data-sources.md)

:::

:::{tab-item} Contributing to Open Climate Risk

If you want to **develop or contribute**:

1. Follow the [Installation](how-to/installation.md) guide to set up your development environment
2. Read the [Project Structure](reference/project-structure.md) to understand the codebase
3. Review [Snapshot Testing](how-to/snapshot-testing.md) for testing practices

:::

::::

## Support

-   **Issues & Bug Reports**: [GitHub Issues](https://github.com/carbonplan/ocr/issues)
-   **General Inquiries**: [hello@carbonplan.org](mailto:hello@carbonplan.org)

## License

Open Climate Risk code is released under the MIT License. See [LICENSE](https://github.com/carbonplan/ocr/blob/main/LICENSE) for details. See [Access Data](access-data.md) for information about data licensing.

```{toctree}
:hidden:
:maxdepth: 2
access-data
terms-of-data-access
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Methods

methods/fire-risk/overview
methods/fire-risk/horizontal-scaling-via-spatial-chunking
methods/fire-risk/score-bins
methods/fire-risk/evaluation
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/api
reference/project-structure
reference/deployment
reference/data-schema
reference/data-sources
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: How-to guides

how-to/user-guide-index
how-to/developer-guide-index
```
