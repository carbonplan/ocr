# Open Climate Risk

**Open Climate Risk** is CarbonPlan's platform for analyzing and visualizing climate-related risks at a building-level resolution across the continental United States (CONUS).

## Quick links

- [**Map tool**](https://carbonplan.org/research/climate-risk)
- [**Explainer article**](https://carbonplan.org/research/climate-risk-explainer)
- [**GitHub repository**](https://github.com/carbonplan/ocr)
- [**Dataset releases**](https://github.com/carbonplan/ocr/releases)

## Getting started

::::{tab-set}

:::{tab-item} Using Open Climate Risk data

If you want to **access and analyze fire risk data**:

1. Visit [Access data](access-data.md) for an overview of all available formats.
2. Read the [Work with data](how-to/work-with-data.ipynb) guide to learn how to load production datasets.
3. Explore the [Data schema](reference/data-schema.md) to understand available variables.

:::

:::{tab-item} Understanding methods

If you want to **understand how fire risk is calculated**:

1. Read the [Fire risk overview](methods/fire-risk/overview.md) for conceptual background.
2. Learn about how we performed [Evaluation](methods/fire-risk/evaluation.md) of our fire risk estimates.
3. Review our [Data sources](reference/data-sources.md).

:::

:::{tab-item} Contributing to Open Climate Risk

If you want to **develop or contribute**:

1. Follow the [Installation](how-to/installation.md) guide to set up your development environment.
2. Read the [Project structure](reference/project-structure.md) to understand the codebase.
3. Review [Snapshot testing](how-to/snapshot-testing.md) for testing practices/

:::

::::

## Support

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/carbonplan/ocr/issues)
- **General Inquiries**: [hello@carbonplan.org](mailto:hello@carbonplan.org)

## License

Open Climate Risk code is released under the MIT License. See [LICENSE](https://github.com/carbonplan/ocr/blob/main/LICENSE) for details. See [Access data](access-data.md) for information about data licensing.

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
