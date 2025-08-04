# OCR Data Pipeline

![](../assets/ocr_data_flow.png)

The OCR (Optimal Climate Risk) data pipeline processes climate risk data through a series of coordinated stages, from individual region processing to final tile generation for visualization.

## Overview

The pipeline transforms raw climate data into risk assessments through four main stages:

1. **Region Processing** - Calculate risk metrics for individual geographic regions
2. **Data Aggregation** - Combine regional results into consolidated datasets
3. **Statistical Summaries** - Generate county and tract-level statistics (optional)
4. **Tile Generation** - Create PMTiles for web visualization

## Getting Started

### Prerequisites

- Python environment with OCR package installed (see [installation guide](../getting-started/installation.md))
- AWS credentials (for data access)
- Coiled account (for cloud execution, optional)

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
OCR_ICECHUNK_STORE_URI=s3://your-bucket/icechunk-store
OCR_VECTOR_OUTPUT_URI=s3://your-bucket/vector-output
OCR_BRANCH=main

# Optional: Coiled credentials (for cloud execution)
COILED_API_TOKEN=your_coiled_token
```

Use your configuration file:

```bash
ocr run --env-file .env --region-id y10_x2
```

### Key Configuration Components

- **Icechunk Store** - Version-controlled data storage backend
- **Vector Output** - Location for processed geoparquet and PMTiles files
- **Branch** - Data version/environment (prod, QA, etc.)
- **Chunking** - Defines valid region boundaries and IDs

## CLI Commands

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
- `--summary-stats` - Include regional statistical summaries
- `--debug` - Enable detailed logging

**Examples:**

```bash
# Development workflow
ocr run --region-id y10_x2 --platform local --debug

# Production processing with statistics
ocr run --all-region-ids --platform coiled --summary-stats --env-file prod.env

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

#### `ocr aggregate` - Data Consolidation

Combine processed regional geoparquet files into a unified dataset.

```bash
ocr aggregate --env-file .env
```

#### `ocr aggregate-regional-risk` - Statistical Summaries

Generate county and tract-level risk statistics.

```bash
ocr aggregate-regional-risk --env-file .env
```

#### `ocr create-regional-pmtiles` - Regional Tiles

Create PMTiles for county and tract-level visualizations.

```bash
ocr create-regional-pmtiles --env-file .env
```

#### `ocr create-pmtiles` - Primary Tiles

Generate PMTiles from the main consolidated dataset.

```bash
ocr create-pmtiles --env-file .env
```

## Troubleshooting

### Common Issues

**Environment Configuration Issues**

```
Error: Missing required environment variables
```

**Solutions:**

- Verify `.env` file exists and is properly formatted
- Check all required AWS credentials are set
- Ensure Coiled credentials are configured (for cloud platform)

**Resource and Access Issues**

_Local Platform:_

- **Disk space:** Check available space in temp directory
- **Memory:** Reduce dataset size or increase system RAM
- **Permissions:** Verify file/directory access rights

_Coiled Platform:_

- **Job failures:** Check Coiled credentials and account quotas
- **AWS access:** Verify IAM permissions and credentials
- **Network:** Confirm AWS region and connectivity
