# Open Climate Risk Data Pipeline

The Open Climate Risk (OCR) data pipeline processes climate risk data through a series of coordinated stages, from individual region processing to final tile generation for visualization.

## Overview

The pipeline transforms raw climate data into risk assessments through four main stages:

1. **Region Processing** - Calculate risk metrics for individual geographic regions
2. **Data Aggregation** - Combine regional results into consolidated datasets
3. **Statistical Summaries** - Generate county and tract-level statistics (optional)
4. **Tile Generation** - Create PMTiles for web visualization

## Getting Started

### Prerequisites

- Python environment with OCR package installed (see [installation guide](../how-to/installation.md))
- AWS credentials (for data access)
- Coiled account (for cloud execution, optional)

### Tutorial: quick end-to-end (local)

This tutorial walks you through a short, practical run that processes one region locally and inspects the output.

1. Ensure your environment is configured and the package is installed (see [installation guide](../how-to/installation.md)).
1. Copy an example env and set a local storage path for quick testing:

```bash
cp ocr-local.env .env
# For local testing you can set OCR_STORAGE_ROOT to a local path, e.g. ./output/
```

1. Run a single-region processing job locally:

```bash
ocr process-region y10_x2 --risk-type fire --platform local
```

1. Inspect outputs in the storage root (geoparquet files and logs):

```bash
ls -la $OCR_STORAGE_ROOT/
```

1. If you set `OCR_DEBUG=1` you will see detailed logs printed to stdout.

### Tutorial: quick end-to-end (Coiled)

Use Coiled for parallel, large-scale processing.

1. Ensure Coiled credentials are set by logging into your account via the Coiled CLI.
2. Run an example multi-region job on Coiled:

```bash
ocr run --region-id y10_x2 --region-id y11_x3 --platform coiled --env-file .env
```

1. Monitor the job on Coiled's web UI and check outputs in your `OCR_STORAGE_ROOT` bucket.

### Basic Usage

Process a single region locally:

```bash
ocr run --region-id y10_x2 --platform local
```

Process multiple regions on Coiled:

```bash
ocr run --region-id y10_x2 --region-id y11_x3 --platform coiled
```

Process all available regions:

```bash
ocr run --all-region-ids --platform coiled
```

## Execution Platforms

### Local Platform

**Best for:** Development, testing, debugging, small datasets

- Runs entirely on your local machine
- Uses local temporary directories
- No cloud costs or dependencies
- Limited by local computational resources
- Sequential processing only

### Coiled Platform

**Best for:** Production workloads, large-scale processing, parallel execution

- Runs on AWS cloud infrastructure
- Automatic resource scaling and management
- Parallel job execution across multiple workers
- Optimized VM types for different workloads
- Built-in monitoring and cost tracking

## Configuration

### Environment Setup

Create a `.env` file for your configuration:

```bash
# .env file for OCR configuration
# OCR Configuration
OCR_STORAGE_ROOT=s3://your-bucket/
OCR_ENVIRONMENT=QA

```

Use your configuration file:

```bash
ocr run --env-file .env --region-id y10_x2
```

### Key Configuration Components

- **Icechunk Store** - Version-controlled data storage backend
- **Vector Output** - Location for processed geoparquet and PMTiles files
- **Environment** - Data version/environment (prod, QA, etc.)
- **Chunking** - Defines valid region boundaries and IDs

## CLI Commands

<!-- prettier-ignore -->
::: mkdocs-click
  :module: ocr.deploy.cli
  :command: ocr
  :prog_name: ocr
  :list_subcommands: true

### Pipeline Orchestration

#### `ocr run` - Full Pipeline

The main command that orchestrates the complete processing pipeline.

**Key Options:**

- `--region-id` - Process specific regions (can specify multiple)
- `--all-region-ids` - Process all available regions
- `--platform` - Choose `local` or `coiled` execution
- `--risk-type` - Calculate `fire` or `wind` risk (default: fire)
- `--write-region-files` - Write regional aggregated summary stats geospatial files.

**Examples:**

```bash
# Development workflow
ocr run --region-id y10_x2 --platform local

# Production processing with statistics
ocr run --all-region-ids --platform coiled --env-file prod.env

# Multi-region wind risk analysis
ocr run --region-id y10_x2 --region-id y11_x3 --risk-type wind --platform coiled

```

### Individual Stage Commands

#### `ocr process-region` - Single Region Processing

Process risk calculations for one specific region.

```bash
# Process fire risk for region y10_x2
ocr process-region y10_x2 --risk-type fire

# Process with custom environment
ocr process-region y15_x7 --env-file production.env --risk-type wind
```

#### `ocr partition-buildings` - Data Consolidation

Partition processed geoparquet files by state and county FIPS codes.

```bash
ocr partition-buildings --env-file .env
```

#### `ocr aggregate-region-risk-summary-stats` - Statistical Summaries

Generate county and tract-level risk statistics.

```bash
ocr aggregate-region-risk-summary-stats --env-file .env
```

#### `ocr create-regional-pmtiles` - Regional Tiles

Create PMTiles for county and tract-level visualizations.

```bash
ocr create-regional-pmtiles --env-file .env
```

#### `ocr create-building-pmtiles` - Building PMTiles

Generate PMTiles from the consolidated building dataset.

```bash
ocr create-building-pmtiles --env-file .env
```

#### `ocr write-aggregated-region-analysis-files` - Write Analysis Files

Write aggregated region analysis files (csv, geoparquet and geojson).
You can add the flag `--write-region-files` to `ocr run` to add this optional step in the pipeline.

```bash
ocr write-aggregated-region-analysis-files --env-file .env
```

## Troubleshooting

### Common Issues

### Environment configuration issues

```text
Error: Missing required environment variables
```

**Solutions:**

- Verify `.env` file exists and is properly formatted
- Check all required AWS credentials are set
- Ensure Coiled credentials are configured (for cloud platform)

### Resource and access issues

#### Local Platform

- **Disk space:** Check available space in temp directory
- **Memory:** Reduce dataset size or increase system RAM
- **Permissions:** Verify file/directory access rights

#### Coiled Platform

- **Job failures:** Check Coiled credentials and account quotas
- **AWS access:** Verify IAM permissions and credentials
- **Network:** Confirm AWS region and connectivity
