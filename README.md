# Open Climate Risk (OCR) Platform

| CI          | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![Deploy Status][github-deploy-badge]][github-deploy-link] [![Code Coverage Status][codecov-badge]][codecov-link] [![pre-commit.ci status][pre-commit.ci-badge]][pre-commit.ci-link] |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **License** |                                                                                                       [![License][license-badge]][repo-link]                                                                                                       |

A scalable pipeline for calculating climate risk assessments at building-level resolution across the continental United States. OCR processes wildfire and wind risk data through a distributed processing system that can run locally or on cloud infrastructure.

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

### Local Development Workflow

The project uses dotenv-style env files. Example files in the repo include [`ocr-local.env`](ocr-local.env) and [`ocr-coiled-s3.env`](ocr-coiled-s3.env) — copy one of these to `.env` and edit values as needed.

Important environment variables:

- `OCR_STORAGE_ROOT`: S3 path or local path where output is written (e.g. `s3://your-bucket/`).
- `OCR_ENVIRONMENT`: name of the environment (e.g. `qa`, `staging`, `production`).
- `OCR_DEBUG`: set to `1` to enable more verbose logging for local troubleshooting.

Start a dev shell with the project environment (we use `pixi`):

```bash
pixi shell
```

```bash
Run minimal pipeline locally
ocr run --region-id y10_x2 --platform local --env-file .env
```

## Contributing

We welcome contributions! Please see our [contributing guide](./contributing.md) for detailed instructions on:

- Setting up your development environment
- Running tests and quality checks
- Submitting pull requests
- Code style and standards

## Support

- **Documentation** - [Full documentation](https://carbonplan-ocr.readthedocs.io)
- **Issues** - Report bugs or request features via [GitHub Issues](https://github.com/carbonplan/ocr/issues)
- **Discussions** - Join the conversation in [GitHub Discussions](https://github.com/carbonplan/ocr/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
