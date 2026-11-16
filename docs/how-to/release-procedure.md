# Release procedure

This guide shows how to release a new version and deploy its data. Publishing a release no longer triggers a deploy: the GitHub Actions workflow is retired, and a CI job that runs `ocr run` is not built yet. Deploy by hand with `ocr run`.

## Prerequisites

- Write access to the repository
- Understanding of [semantic versioning](https://semver.org/)
- Changes merged to `main`
- A run file per risk for the version, `runs/<risk>/vX.Y.Z.toml` (see [Data pipeline](data-pipeline.md))

## Steps

### 1. Publish the release

[release-drafter](https://github.com/release-drafter/release-drafter) keeps a draft release up to date as PRs merge to `main`.

1. Open the draft at the top of the repository's Releases page
2. Check the auto-generated notes: every merged PR listed, categories right
3. Click **Edit draft** and set:
    - **Tag**: the next version, e.g. `v1.2.3` (patch: fixes; minor: backward-compatible features; major: breaking changes)
    - **Title**: `vX.Y.Z`, optionally with a description
    - **Notes**: migration notes, breaking changes, known issues
    - **Target**: `main`
4. Click **Publish release**

### 2. Deploy the data

From a checkout of the release tag, per risk:

```bash
ocr run runs/<risk>/vX.Y.Z.toml --dry-run
ocr run runs/<risk>/vX.Y.Z.toml
```

`--dry-run` checks the stages, kwargs and inputs without starting any job. A failed or interrupted run resumes when rerun; `ocr run --help` lists the options.

### 3. Verify the outputs

Outputs land under `s3://carbonplan-ocr/output/<risk>/vX.Y.Z/` (`ocr.config.OutputPaths`):

```bash
aws s3 ls s3://carbonplan-ocr/output/<risk>/vX.Y.Z/
```

### 4. Deploy to the web tool

1. Bump the version in `lib/config.ts` of the [web tool repository](https://github.com/carbonplan/ocr-web)
2. Open a PR to the `production` branch
3. Check the data in the Vercel preview linked on the PR
4. Merge. Vercel deploys to <https://carbonplan.org/research/climate-risk>

## Reference

- **Production website**: <https://carbonplan.org/research/climate-risk>
- **QA URL**: <https://ocr.qa.carbonplan.org>
- **Run files and stages**: [Data pipeline](data-pipeline.md)
