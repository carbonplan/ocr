# Welcome to OCR Documentation

**Open Climate Risk (OCR)** is CarbonPlan's platform for analyzing and visualizing climate-related risks across the United States.... [TK]

## What is OCR?

TODO: Add description of OCR here. [TK]

## Quick Links

- **🌐 Live Web Tool**: [ocr.carbonplan.org](https://ocr.carbonplan.org)
- **📰 Explainer Article**: [Explainer Article](TK)
- **💻 GitHub Repository**: [carbonplan/ocr](https://github.com/carbonplan/ocr)
- **📊 Data Access**: [S3 Bucket](./how-to/work-with-data.ipynb)
- **📄 Changelog**: [Changelog](https://github.com/carbonplan/ocr/releases)

## Getting Started

=== "Using OCR Data"

    If you want to **access and analyze OCR's fire risk data**:

    1. Start with [Working with Data](how-to/work-with-data.ipynb) to learn how to load production datasets
    2. Explore the [Data Schema](reference/data-schema.md) to understand available variables
    3. Check [Data Downloads](reference/data-downloads.md) for direct S3 access

=== "Contributing to OCR"

    If you want to **develop or contribute to OCR**:

    1. Follow the [Installation](how-to/installation.md) guide to set up your development environment
    2. Read the [Project Structure](reference/project-structure.md) to understand the codebase
    3. Review [Snapshot Testing](how-to/snapshot-testing.md) for testing practices

=== "Understanding Methods"

    If you want to **understand how fire risk is calculated**:

    1. Read the [Fire Risk Overview](methods/fire-risk/overview.md) for conceptual background
    2. Explore [Data Sources and Provenance](methods/fire-risk/data-sources-and-provenance.md)
    3. Review [Implementation](methods/fire-risk/implementation.ipynb) for technical details

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
- **Discussions**: [GitHub Discussions](https://github.com/carbonplan/ocr/discussions)
- **General Inquiries**: [hello@carbonplan.org](mailto:hello@carbonplan.org)

## Citation (TK)

If you use OCR data or methods in your research, please cite:

```bibtex
@software{carbonplan_ocr,
  author = {CarbonPlan},
  title = {Open Climate Risk (OCR)},
  year = {2025},
  url = {https://github.com/carbonplan/ocr},
  note = {Version X.Y.Z}
}
```

## License (TK)

OCR code is released under the MIT License. See [LICENSE](https://github.com/carbonplan/ocr/blob/main/LICENSE) for details.

---

_Ready to get started? Pick a path above based on what you want to do with OCR!_
