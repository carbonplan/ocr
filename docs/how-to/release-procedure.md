# Release to Production

This guide shows how to release a new version of the project to production.

## Prerequisites

- Write access to the [carbonplan/ocr](https://github.com/carbonplan/ocr) repository
- Understanding of [semantic versioning](https://semver.org/)
- Changes merged to `main` branch
- Successful staging deployment verified

## Overview

OCR uses [release-drafter](https://github.com/release-drafter/release-drafter) to automatically create draft releases as PRs are merged to `main`. The draft release includes:

- Auto-generated release notes from merged PRs
- Categorized changes (features, bug fixes, etc.)
- List of contributors

**Your job**: Review the draft and publish it.

## Steps

### 1. Verify staging deployment

Before releasing, confirm the staging deployment is working correctly:

```bash
# Check staging environment
open https://ocr.staging.carbonplan.org
```

Verify that:

- Data loads correctly
- No errors in processing logs
- All expected regions are present

### 2. Review the draft release

1. Go to <https://github.com/carbonplan/ocr/releases>
2. Find the draft release at the top (usually labeled "Next Release")
3. Review the auto-generated release notes:
    - Check that all merged PRs are listed
    - Verify categorization is correct
    - Ensure contributor list is accurate

### 3. Edit release details

Click **Edit draft** and update:

1. **Tag version**: Set to next semantic version (e.g., `v1.2.3`)
    - **Patch** (`1.2.3` → `1.2.4`): Bug fixes, minor changes
    - **Minor** (`1.2.3` → `1.3.0`): New features, backward compatible
    - **Major** (`1.2.3` → `2.0.0`): Breaking changes

2. **Release title**: Use format `vX.Y.Z` or add description (e.g., `v1.2.3 - Fire Risk Updates`)

3. **Release notes**: Edit the auto-generated notes if needed:
    - Add any important context or migration notes
    - Highlight breaking changes (if major version)
    - Note any known issues or limitations

4. **Target branch**: Ensure it's set to `main`

### 4. Publish the release

1. Review everything one final time
2. Click **Publish release**

Publishing automatically creates the tag on `main` and triggers production deployment.

### 5. Monitor automated deployment

Publishing the release triggers the production deployment automatically.

**What happens automatically:**

1. **Tag creation**: GitHub creates tag `vX.Y.Z` on `main` branch
2. **Coiled environment creation** (~5-10 min)
    - Builds software environment named `ocr-vX.Y.Z`

3. **Production deployment** (~2-4 hours)
    - Processes all regions across CONUS
    - Generates aggregated statistics
    - Creates PMTiles for visualization
    - Writes output to `s3://carbonplan-ocr/output/fire-risk/tensor/production/vX.Y.Z/`

**Monitor progress:**

```bash
# View GitHub Actions workflow
open https://github.com/carbonplan/ocr/actions/workflows/deploy.yaml

# Check Coiled dashboard
open https://cloud.coiled.io/
```

### 6. Verify production deployment

After deployment completes (check Actions page for ✓):

```bash
# Check production environment
open https://carbonplan.org/research/climate-risk
```

Verify:

- Version matches your release tag
- Data is accessible
- Map tiles load correctly
- Regional statistics are complete

## What the automated workflow does

**Release Drafter** (runs on every merge to `main`):

- Automatically creates/updates a draft release
- Generates release notes from PR titles and labels
- Categorizes changes (features, fixes, documentation, etc.)
- Lists all contributors

**Production Deployment** (runs when you publish a release):

1. **Validates version**: Ensures tag is valid semantic version (e.g., `1.2.3`)
2. **Sets environment variables**:
    - `OCR_VERSION=X.Y.Z`
    - `OCR_ENVIRONMENT=production`
3. **Updates configuration**: Injects version into `ocr-coiled-s3-production.env`
4. **Runs full pipeline**:
    - Processes all regions (`--all-region-ids`)
    - Writes regional statistics
    - Generates regional PMTiles
    - Creates aggregated outputs
5. **Deploys to S3**: Outputs stored at versioned path

## Manual production redeployment

If you need to redeploy an existing version (e.g., after an intermittent GitHub or Coiled failure):

1. Go to <https://github.com/carbonplan/ocr/actions/workflows/deploy.yaml>
2. Click **Run workflow** dropdown
3. Select branch: `main`
4. Fill in parameters:
    - **production_tag**: Enter the existing version tag to redeploy (e.g., `v1.2.3` or `1.2.3`)
    - Leave other fields at defaults (they're ignored for production redeploy)
5. Click **Run workflow**

This redeploys the exact code from that existing release tag to production.

**Note**: This is only for redeploying existing releases. To create a new release, follow the main steps above.

## Troubleshooting

### Deployment fails during region processing

**Symptoms**: Some regions fail to process

**Solution**: The workflow automatically retries failed regions (5 attempts with backoff). Check logs for persistent failures.

### Version validation error

**Symptoms**: "Not valid SemVer" error

**Solution**: Ensure tag follows format `vX.Y.Z` or `X.Y.Z` where X, Y, Z are numbers.

### Deployment takes longer than expected

**Symptoms**: Workflow runs beyond 4 hours

**Solution**: This is expected for full CONUS processing. Check:

- Coiled cluster is running
- No AWS credential issues
- S3 buckets are accessible

### Wrong version deployed

**Symptoms**: Production shows incorrect version

**Solution**:

1. Check which tag was used in the deployment
2. Redeploy correct version using manual workflow (see above)

## Reference

- **Production URL**: <https://carbonplan.org/research/climate-risk>
- **Staging URL**: <https://ocr.staging.carbonplan.org>
- **Workflow file**: `.github/workflows/deploy.yaml`
- **Deployment details**: See [deployment reference](../reference/deployment.md)

## Next steps

After releasing:

- Monitor production for any issues
- Update changelog or documentation if needed
