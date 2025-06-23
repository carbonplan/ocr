# Contributing Guide

Contributions to this project are welcome! If you're new to contributing to CarbonPlan projects, here are some resources to help you get started:

- Familiarize yourself with our [GitHub organization](https://github.com/carbonplan)
- Read our [Code of Conduct](https://github.com/carbonplan/.github/blob/main/CODE_OF_CONDUCT.md)

## Contributing code

We follow a standard Github-centric model for collaborating on software projects. Here are the basic steps:

### 1. Fork the repository

Click the "Fork" button in the top right corner of the repository page to create your own copy of the repository. Once you have forked the repository, clone it to your local machine:

```bash
git clone https://github.com/{user-name}/{repo-name}.git
```

### 2. Create a development environment

We use [`pixi`](https://pixi.sh/latest/) to manage our development environment. To set it up, please follow the instructions from pixi's [installation guide](https://pixi.sh/latest/). Once pixi is installed, you can create local development environments for the project by running:

```bash
pixi install
```

This will install all the necessary dependencies for the project. Once the installation is complete, you can enable `pre-commit` hooks to ensure that your code adheres to our style guidelines:

```bash
pixi pre-commit install
```

This will set up pre-commit hooks that will automatically format your code and run tests before you commit any changes.

### 3. Create a new branch

Before you get started with your work on the project, it's a good idea to create a new branch. This keeps your changes organized and makes it easier to submit a pull request later. You can create a new branch with the following command:

```bash
git fetch upstream
git checkout -b {branch-name} upstream/main
```

Replace `{branch-name}` with a descriptive name for your branch.

### 4. Run tests

Before you start making changes, it's a good idea to run the tests to ensure everything is working as expected. You can run the tests with the following command:

```bash
pixi run tests
```

This will execute the test suite and let you know if everything is functioning correctly. If you are adding new features or making changes, consider adding tests for your new code as well.

### 5. Commit your changes

Then, once you have made your changes, you can commit them to your branch:

```bash
git add .
git commit
git push origin {branch-name}
```

### 6. Submit a pull request

Once you have pushed your changes to your forked repository, you can submit a pull request to the original repository. Go to the original repository on GitHub, and you should see a prompt to create a pull request for your branch. Click on it and fill out the necessary details. Make sure to provide a clear description of the changes you made and why they are necessary.

## Contributing documentation

!Note: Directions for uv, we can update to pixi.

Our documentation is built with `mkdocs`.

To build the docs:

1.

```
uv run --group docs mkdocs serve
...

2. Navigate to localhost:8000
```
