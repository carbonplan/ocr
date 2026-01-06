# CONUS404 Data Processing Pipeline

This directory contains the end-to-end pipeline for processing [CONUS404](https://www.sciencebase.gov/catalog/item/6372cd09d34ed907bf6c6ab1) meteorological data for fire risk analysis. The pipeline consists of two main stages:

1. **Variable Subsetting**: Download and rechunk individual meteorological variables from Open Storage Network (OSN)
2. **FFWI Computation**: Calculate Fosberg Fire Weather Index with quantiles, wind direction distributions, and reprojection

> **Note**: The original scripts (`subset_conus404_from_osn.py` and `compute_fosberg_fire_weather_index.py`) have been migrated to the unified OCR input dataset CLI. See the deprecation notices in those files for migration information.

## Quick Start

All processing is now done through the unified `ocr ingest-data` CLI:

```bash
# List all available datasets
pixi run ocr ingest-data list-datasets

# Get help for CONUS404 processing
pixi run ocr ingest-data process conus404-subset --help
pixi run ocr ingest-data process conus404-ffwi --help
```

## Stage 1: Variable Subsetting

### Overview

Downloads individual CONUS404 meteorological variables from OSN and rechunks them from temporal+spatial chunks to spatial-only chunks for efficient spatial queries.

**Available variables**: `Q2`, `TD2`, `PSFC`, `T2`, `V10`, `U10`

### Commands

```bash
# Dry-run to preview operations
pixi run ocr ingest-data process conus404-subset \
  --conus404-variable Q2 \
  --dry-run

# Process a single variable (Q2)
pixi run ocr ingest-data process conus404-subset \
  --conus404-variable Q2 \
  --use-coiled \
  --software <your-coiled-env>

# Process with custom spatial tile size
pixi run ocr ingest-data process conus404-subset \
  --conus404-variable U10 \
  --conus404-spatial-tile-size 10 \
  --use-coiled \
  --software <your-coiled-env>

# Process all variables (run this command 6 times, once per variable)
for var in Q2 TD2 PSFC T2 V10 U10; do
  pixi run ocr ingest-data process conus404-subset \
    --conus404-variable $var \
    --use-coiled \
    --software <your-coiled-env>
done
```

### Configuration

-   **Workers**: 15 (configurable via class constant `COILED_WORKERS`)
-   **Worker VM**: `r6a.8xlarge` (configurable via `COILED_WORKER_VM`)
-   **Scheduler VM**: `c6a.large` (configurable via `COILED_SCHEDULER_VM`)
-   **Spatial tile size**: 10 (adjustable via `--conus404-spatial-tile-size`)
-   **Output**: `s3://carbonplan-ocr/input/conus404-hourly-icechunk/{variable}`

## Accessing the dataset

The processed dataset is stored in the `carbonplan-ocr` bucket on S3. You can access it using the following code snippet:

```python
In [1]: import xarray as xr

In [2]: import icechunk

In [3]: variables = ['Q2', 'TD2', 'PSFC', 'T2', 'V10', 'U10']
   ...: stores = []
   ...: for variable in variables:
   ...:     config = {
   ...:     'bucket': 'carbonplan-ocr',
   ...:     'prefix': f'input/conus404-hourly-icechunk/{variable}',
   ...:     'region': 'us-west-2',
   ...: }
   ...:     storage = icechunk.s3_storage(
   ...:     bucket=config['bucket'],
   ...:     prefix=config['prefix'],
   ...:     region=config['region'],
   ...: )
   ...:     repo = icechunk.Repository.open(storage)
   ...:     session = repo.readonly_session('main')
   ...:     stores.append(session.store)
   ...:
   ...:
   ...:

In [4]: ds = xr.open_mfdataset(stores, engine='zarr', consolidated=False, chunks={}, parallel=True)

In [5]: ds
Out[5]:
<xarray.Dataset> Size: 13TB
Dimensions:  (time: 376945, y: 1015, x: 1367)
Coordinates:
    lat      (y, x) float32 6MB dask.array<chunksize=(10, 10), meta=np.ndarray>
  * x        (x) float64 11kB -2.732e+06 -2.728e+06 ... 2.728e+06 2.732e+06
  * time     (time) datetime64[ns] 3MB 1979-10-01 ... 2022-10-01
  * y        (y) float64 8kB -2.028e+06 -2.024e+06 ... 2.024e+06 2.028e+06
    lon      (y, x) float32 6MB dask.array<chunksize=(10, 10), meta=np.ndarray>
Data variables:
    Q2       (time, y, x) float32 2TB dask.array<chunksize=(376945, 10, 10), meta=np.ndarray>
    TD2      (time, y, x) float32 2TB dask.array<chunksize=(376945, 10, 10), meta=np.ndarray>
    PSFC     (time, y, x) float32 2TB dask.array<chunksize=(376945, 10, 10), meta=np.ndarray>
    T2       (time, y, x) float32 2TB dask.array<chunksize=(376945, 10, 10), meta=np.ndarray>
    V10      (time, y, x) float32 2TB dask.array<chunksize=(376945, 10, 10), meta=np.ndarray>
    U10      (time, y, x) float32 2TB dask.array<chunksize=(376945, 10, 10), meta=np.ndarray>
Attributes: (12/148)
    AER_ANGEXP_OPT:                  1
    AER_ANGEXP_VAL:                  1.2999999523162842
    AER_AOD550_OPT:                  1
    AER_AOD550_VAL:                  0.11999999731779099
    AER_ASY_OPT:                     1
    AER_ASY_VAL:                     0.8999999761581421
    ...                              ...
    WEST-EAST_PATCH_START_STAG:      1
    WEST-EAST_PATCH_START_UNSTAG:    1
    W_DAMPING:                       1
    YSU_TOPDOWN_PBLMIX:              0
    history:                         Tue Mar 29 16:35:22 2022: ncrcat -A -vW ...

```

## Stage 2: Fosberg Fire Weather Index (FFWI)

### FFWI Overview

Computes the Fosberg Fire Weather Index from CONUS404 hourly data with three processing steps:

1. **compute**: Calculate base FFWI and winds from relative humidity, temperature, and wind speed
2. **postprocess**: Compute quantiles (e.g., 99th percentile) and wind direction distributions over time
3. **reproject**: Reproject wind direction distribution to EPSG:4326 geobox

### FFWI Workflow Steps

#### Step 1: Compute Base FFWI

```bash
# Dry-run to preview
pixi run ocr ingest-data process conus404-ffwi \
  --ffwi-processing-step compute \
  --dry-run

# Run with Coiled
pixi run ocr ingest-data process conus404-ffwi \
  --ffwi-processing-step compute \
  --use-coiled \
  --software <your-coiled-env>
```

This step:

-   Loads CONUS404 variables with spatial constants
-   Computes relative humidity from specific humidity and temperature
-   Rotates grid-relative winds to earth coordinates
-   Computes FFWI using `fosberg_fire_weather_index(hurs, T2, sfcWind)`
-   Writes base FFWI and wind fields to Icechunk

Output:

-   `s3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index.icechunk`
-   `s3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi/winds.icechunk`

#### Step 2: Postprocess Quantiles and Wind Direction Distribution

```bash
# Compute 99th percentile (default)
pixi run ocr ingest-data process conus404-ffwi \
  --ffwi-processing-step postprocess \
  --use-coiled \
  --software <your-coiled-env>

# Compute multiple quantiles
pixi run ocr ingest-data process conus404-ffwi \
  --ffwi-processing-step postprocess \
  --ffwi-quantiles 0.99 \
  --ffwi-quantiles 0.95 \
  --use-coiled \
  --software <your-coiled-env>
```

This step:

-   Loads base FFWI from step 1 (validates prerequisite exists)
-   Computes time quantiles for each spatial location
-   Computes wind direction distribution during high-FFWI hours (hours where FFWI >= quantile threshold)
-   Assigns cardinal direction labels (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`) to wind direction dimension

Output:

-   `s3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index-p99.icechunk`
-   `s3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index-p99-wind-direction-distribution.icechunk`

#### Step 3: Reproject Wind Direction Distribution

```bash
pixi run ocr ingest-data process conus404-ffwi \
  --ffwi-processing-step reproject \
  --use-coiled \
  --software <your-coiled-env>
```

This step:

-   Loads wind direction distribution from step 2 (validates prerequisite exists)
-   Reprojects to EPSG:4326 geobox matching Scott et al. 2024 dataset (30m resolution)
-   Uses nearest neighbor resampling for categorical wind direction data

Output:

-   `s3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index-p99-wind-direction-distribution-30m-4326.icechunk`

### Run All Steps

```bash
# Run entire FFWI pipeline (compute → postprocess → reproject)
pixi run ocr ingest-data process conus404-ffwi \
  --ffwi-processing-step all \
  --ffwi-quantiles 0.99 \
  --use-coiled \
  --software <your-coiled-env>
```

### FFWI Configuration

-   **Compute workers**: 10 (class constant `COILED_WORKERS_COMPUTE`)
-   **Postprocess workers**: 10 (class constant `COILED_WORKERS_POSTPROCESS`)
-   **Reproject workers**: 10 (class constant `COILED_WORKERS_REPROJECT`)
-   **Worker VM**: `m8g.2xlarge`
-   **Scheduler VM**: `m8g.large`
-   **Default quantiles**: `[0.99]`

## Accessing Processed Data

### Stage 1: Variable Subsets

```python
import xarray as xr
import icechunk

# Access a single variable
variable = 'Q2'
storage = icechunk.s3_storage(
    bucket='carbonplan-ocr',
    prefix=f'input/conus404-hourly-icechunk/{variable}',
    region='us-west-2',
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session('main')
ds = xr.open_zarr(session.store, chunks={})

# Access all variables at once
variables = ['Q2', 'TD2', 'PSFC', 'T2', 'V10', 'U10']
stores = []
for var in variables:
    storage = icechunk.s3_storage(
        bucket='carbonplan-ocr',
        prefix=f'input/conus404-hourly-icechunk/{var}',
        region='us-west-2',
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session('main')
    stores.append(session.store)

ds = xr.open_mfdataset(stores, engine='zarr', consolidated=False, chunks={}, parallel=True)
# Result: Dataset with dimensions (time: 376945, y: 1015, x: 1367)
```

### Stage 2: FFWI Products

```python
import xarray as xr
import icechunk

base_prefix = 'input/fire-risk/tensor/conus404-ffwi'

# Base FFWI (time series)
storage = icechunk.s3_storage(
    bucket='carbonplan-ocr',
    prefix=f'{base_prefix}/fosberg-fire-weather-index.icechunk',
    region='us-west-2',
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session('main')
ffwi = xr.open_zarr(session.store)

# 99th percentile quantile
storage = icechunk.s3_storage(
    bucket='carbonplan-ocr',
    prefix=f'{base_prefix}/fosberg-fire-weather-index-p99.icechunk',
    region='us-west-2',
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session('main')
ffwi_p99 = xr.open_zarr(session.store)

# Wind direction distribution (with cardinal labels)
storage = icechunk.s3_storage(
    bucket='carbonplan-ocr',
    prefix=f'{base_prefix}/fosberg-fire-weather-index-p99-wind-direction-distribution.icechunk',
    region='us-west-2',
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session('main')
wind_dist = xr.open_zarr(session.store)
# wind_dist has wind_direction coordinate with labels: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

# Reprojected wind direction distribution (EPSG:4326, 30m)
storage = icechunk.s3_storage(
    bucket='carbonplan-ocr',
    prefix=f'{base_prefix}/fosberg-fire-weather-index-p99-wind-direction-distribution-30m-4326.icechunk',
    region='us-west-2',
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session('main')
wind_dist_4326 = xr.open_zarr(session.store)
```

## Complete End-to-End Workflow

```bash
# 1. Process all 6 meteorological variables (can be parallelized)
for var in Q2 TD2 PSFC T2 V10 U10; do
  pixi run ocr ingest-data process conus404-subset \
    --conus404-variable $var \
    --use-coiled \
    --software <your-coiled-env>
done

# 2. Compute FFWI (requires all variables from step 1)
pixi run ocr ingest-data process conus404-ffwi \
  --ffwi-processing-step compute \
  --use-coiled \
  --software <your-coiled-env>

# 3. Postprocess quantiles and wind direction distribution
pixi run ocr ingest-data process conus404-ffwi \
  --ffwi-processing-step postprocess \
  --ffwi-quantiles 0.99 \
  --use-coiled \
  --software <your-coiled-env>

# 4. Reproject wind direction distribution to EPSG:4326
pixi run ocr ingest-data process conus404-ffwi \
  --ffwi-processing-step reproject \
  --use-coiled \
  --software <your-coiled-env>

# Alternative: Run steps 2-4 together
pixi run ocr ingest-data process conus404-ffwi \
  --ffwi-processing-step all \
  --ffwi-quantiles 0.99 \
  --use-coiled \
  --software <your-coiled-env>
```

## Implementation Details

-   **Processor classes**: `Conus404SubsetProcessor` and `Conus404FFWIProcessor` in [ocr/input_datasets/tensor/conus404.py](../../../ocr/input_datasets/tensor/conus404.py)
-   **CLI registration**: [ocr/input_datasets/cli.py](../../../ocr/input_datasets/cli.py)
-   **Fire risk functions**: [ocr/risks/fire.py](../../../ocr/risks/fire.py)
-   **Wind direction labels**: Cardinal and ordinal directions (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`) are assigned at creation time

## Legacy Scripts

The original standalone scripts are deprecated but kept for reference:

-   `subset_conus404_from_osn.py` - See deprecation notice for migration path
-   `compute_fosberg_fire_weather_index.py` - See deprecation notice for migration path
