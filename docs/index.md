# Open Climate Risk

**Open Climate Risk** is CarbonPlan's platform for analyzing and visualizing climate-related risks at a building-level resolution across the Continental United States (CONUS).

## Quick Links

- [**Web Tool**](https://carbonplan.org/research/climate-risk)
- [**Explainer Article**](https://carbonplan.org/research/climate-risk-explainer)
- [**GitHub Repository**](https://github.com/carbonplan/ocr)
- [**Dataset Releases**](https://github.com/carbonplan/ocr/releases)

## Getting Started

=== "Using Open Climate Risk Data"

    If you want to **access and analyze fire risk data**:

    1. Check [Data Downloads](access-data.md) for an overview of all available formats
    2. Start with [Working with Data][access-ocr-output-data] to learn how to load production datasets
    3. Explore the [Data Schema][data-schema] to understand available variables

=== "Contributing to Open Climate Risk"

    If you want to **develop or contribute**:

    1. Follow the [Installation][install-ocr-for-development] guide to set up your development environment
    2. Read the [Project Structure][project-structure] to understand the codebase
    3. Review [Snapshot Testing][snapshot-testing-with-xarrayzarr] for testing practices

=== "Understanding Methods"

    If you want to **understand how fire risk is calculated**:

    1. Read the [Fire Risk Overview][fire-risk-methods-overview] for conceptual background
    2. Learn about how we performed [Evaluation][evaluation] of our fire risk estimates
    2. Review our [Data Sources][data-sources]

## Documentation Structure

This documentation follows the [Diataxis framework](https://diataxis.fr/) to help you quickly find what you need:

<div class="grid cards" markdown>

- :material-school:{ .lg .middle } **Tutorials**

    ***

    Learning-oriented guides that walk you through complete workflows step-by-step.

    [:octicons-arrow-right-24: Browse tutorials](tutorials/data-pipeline.md)

- :material-hammer-wrench:{ .lg .middle } **How-to Guides**

    ***

    Task-oriented instructions to accomplish specific goals, organized by user type.

    [:octicons-arrow-right-24: User guides](how-to/work-with-data.ipynb)
    [:octicons-arrow-right-24: Developer guides](how-to/installation.md)

- :material-book-open-variant:{ .lg .middle } **Methods**

    ***

    Conceptual explanations of our scientific approaches and implementation decisions.

    [:octicons-arrow-right-24: Fire risk methods](methods/fire-risk/overview.md)

- :material-file-document:{ .lg .middle } **Reference**

    ***

    Technical specifications, API documentation, and configuration details.

    [:octicons-arrow-right-24: API reference](reference/api.md)
    [:octicons-arrow-right-24: Deployment](reference/deployment.md)

</div>

## Support

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/carbonplan/ocr/issues)
- **General Inquiries**: [hello@carbonplan.org](mailto:hello@carbonplan.org)

## License

Open Climate Risk code is released under the MIT License. See [LICENSE](https://github.com/carbonplan/ocr/blob/main/LICENSE) for details. See [data downloads](access-data.md) for information about data licensing.
