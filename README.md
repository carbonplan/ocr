# Open Climate Risk (OCR) Platform

| CI          | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![Deploy Status][github-deploy-badge]][github-deploy-link] [![Code Coverage Status][codecov-badge]][codecov-link] [![pre-commit.ci status][pre-commit.ci-badge]][pre-commit.ci-link] |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **License** |                                                                                                       [![License][license-badge]][repo-link]                                                                                                       |
| **Docs**    |                                                                                                   [![Documentation Status][rtd-badge]][rtd-link]                                                                                                   |

A scalable platform for analyzing and visualizing climate-related risks at a building-level resolution across the Continental United States (CONUS)

## Quick Start

### Installation

> [!IMPORTANT]
> We use [`pixi`](https://pixi.sh/latest/) to manage our development environment, which you'll need to install before you can get started with the project.

```bash
# Clone the repository
git clone https://github.com/carbonplan/ocr.git
cd ocr

# Install dependencies with pixi
pixi install

# Activate the environment
pixi shell
```

## Development

### Running Tests

```bash
# Run all tests
pixi run tests

# Run specific test file
pixi run pytest tests/test_datasets.py

# Run integration tests
pixi run tests-integration
```

### Code Quality

```bash
# Format code
pixi run format

# Run linting
pixi run lint
```

We welcome contributions! Please see our [contributing guide](./contributing.md) for detailed instructions on:

-   Setting up your development environment
-   Running tests and quality checks
-   Submitting pull requests

-   **Documentation** - [Full documentation](https://docs.carbonplan.org/ocr)
-   **Issues** - Report bugs or request features via [GitHub Issues](https://github.com/carbonplan/ocr/issues)
-   **Discussions** - Join the conversation in [GitHub Discussions](https://github.com/carbonplan/ocr/discussions)

## Data Access

Our processed input datasets and pipeline outputs can be accessed on [Source Coop](https://source.coop/carbonplan/carbonplan-ocr).
Details on how to use the data can be found in our [technical documentation](https://carbonplan.github.io/ocr/how-to/work-with-data/).

## License

> [!IMPORTANT]
> OCR code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. See the [data access page](./docs/access-data.md) for information on the licenses for the various datasets.

[github-ci-badge]: https://github.com/carbonplan/ocr/actions/workflows/ci.yaml/badge.svg
[github-ci-link]: https://github.com/carbonplan/ocr/actions/workflows/ci.yaml
[github-deploy-badge]: https://github.com/carbonplan/ocr/actions/workflows/deploy.yaml/badge.svg
[github-deploy-link]: https://github.com/carbonplan/ocr/actions/workflows/deploy.yaml
[codecov-badge]: https://img.shields.io/codecov/c/github/carbonplan/ocr.svg?logo=codecov
[codecov-link]: https://codecov.io/gh/carbonplan/ocr
[license-badge]: https://img.shields.io/github/license/carbonplan/ocr
[repo-link]: https://github.com/carbonplan/ocr
[pre-commit.ci-badge]: https://results.pre-commit.ci/badge/github/carbonplan/ocr/main.svg
[pre-commit.ci-link]: https://results.pre-commit.ci/latest/github/carbonplan/ocr/main
[rtd-badge]: https://readthedocs.org/projects/open-climate-risk/badge/?version=latest
[rtd-link]: https://open-climate-risk.readthedocs.io/en/latest/?badge=latest
