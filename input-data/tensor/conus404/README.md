# A subset of the CONUS-404 dataset

The `subset_conus404_from_osn.py` Python module contains code used to subset and rechunk the [CONUS404](https://www.sciencebase.gov/catalog/item/6372cd09d34ed907bf6c6ab1) dataset. The subset is created from the original dataset available from the open storage network (OSN). The original dataset is chunked along the sptial and temporal dimensions. We transferred a subset of this dataset to our bucket and rechunked it to create a version that is more suitable for our analysis. This new version is chunked along the spatial dimensions only.

## Usage

To see the available options, you can run the following command:

```bash
pixi run python input-data/tensor/conus404/subset_conus404_from_osn.py --help                                                                                                       ─╯

 Usage: subset_conus404_from_osn.py [OPTIONS] VARIABLE

 Main entry point for processing CONUS-404 dataset.


╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    variable      TEXT  variable to process from the dataset [default: None] [required]                                                                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --worker-vm-type            TEXT     VM type for worker nodes in the Coiled cluster [default: r6a.8xlarge]                                                                            │
│ --scheduler-vm-type         TEXT     VM type for the scheduler node in the Coiled cluster [default: c6a.large]                                                                        │
│ --n-workers                 INTEGER  Number of worker nodes in the Coiled cluster [default: 15]                                                                                       │
│ --spatial-tile-size         INTEGER  Size of spatial tiles for chunking [default: 10]                                                                                                 │
│ --install-completion                 Install completion for the current shell.                                                                                                        │
│ --show-completion                    Show completion for the current shell, to copy it or customize the installation.                                                                 │
│ --help                               Show this message and exit.                                                                                                                      |
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

You can run it on coiled with the following command to process a specific variable, for example `U10`:

```bash
pixi run coiled batch run input-data/tensor/conus404/subset_conus404_from_osn.py U10
```

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

## Fosberg Fire Weather Index (FFWI)

The script `compute_fosberg_fire_weather_index.py` computes the Fosberg Fire Weather Index from CONUS404 hourly data, writes the base FFWI field to an Icechunk repo, and postprocesses quantiles and prevailing wind direction during high-FFWI periods. A utility command reprojects the mode product to the geobox of a catalog dataset.

### What it does (FFWI)

- Loads CONUS404 with spatial constants via `load_conus404(add_spatial_constants=True)`.
- Computes relative humidity from specific humidity and temperature.
- Rotates grid-relative winds to earth-relative components and derives wind speed and direction.
- Computes FFWI using `fosberg_fire_weather_index(hurs, T2, sfcWind)` and writes to Icechunk.
- Saves the derived surface wind fields used in the computation.
- Postprocesses: time-quantiles of FFWI and wind-direction distribution/mode where FFWI exceeds those quantiles.
- Reprojects the mode dataset to the `USFS-wildfire-risk-communities-4326` geobox.

### Commands

The module is a Typer app with three subcommands.

1. Compute base FFWI and winds

```bash
pixi run python input-data/tensor/conus404/compute_fosberg_fire_weather_index.py compute --help
```

Key options:

- `--dry-run`: Use a tiny spatial slice and skip cluster startup.
- `--overwrite`: Overwrite existing Icechunk data.
- `--output-base`: Base path for outputs (S3 or local). Default: `s3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi`.
- `--min-workers`, `--max-workers`, `--worker-vm-types`: Coiled autoscaling and instance types.

Example (Coiled):

```bash
pixi run coiled batch run input-data/tensor/conus404/compute_fosberg_fire_weather_index.py compute \
  --min-workers 10 --max-workers 70 --worker-vm-types m8g.2xlarge
```

1. Postprocess quantiles and mode

```bash
pixi run python input-data/tensor/conus404/compute_fosberg_fire_weather_index.py postprocess --help
```

Key options:

- `--quantiles`: List of quantiles to compute (default `[0.95, 0.99]`).
- `--mode/--no-mode`: Compute prevailing wind direction for hours where FFWI exceeds the quantile.
- Same cluster options as above.

Example:

```bash
pixi run coiled batch run input-data/tensor/conus404/compute_fosberg_fire_weather_index.py postprocess \
  --quantiles 0.95 0.99 --mode True
```

1. Reproject FFWI mode

```bash
pixi run python input-data/tensor/conus404/compute_fosberg_fire_weather_index.py reproject-mode --help
```

Reprojects the mode dataset to the geobox for `USFS-wildfire-risk-communities-4326`.

### Outputs (FFWI)

Under the base `--output-base` (default `s3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi`):

- `fosberg-fire-weather-index.icechunk` — Base FFWI over time (x, y, time).
- `winds.icechunk` — Derived surface wind speed and direction fields.
- `fosberg-fire-weather-index_p95.icechunk`, `fosberg-fire-weather-index_p99.icechunk` — Time-quantiles of FFWI (x, y).
- `fosberg-fire-weather-index_p95_wind_direction_distribution.icechunk` — Per-direction histogram for hours exceeding the quantile.
- `fosberg-fire-weather-index_p95_mode.icechunk` — Mode of wind direction during high-FFWI hours.
- `fosberg-fire-weather-index_p99_mode_reprojected.icechunk` — Mode reprojected to target geobox (when using `reproject-mode`).

### Reading results

```python
import xarray as xr

base = 's3://carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi'
ffwi = xr.open_zarr(f'{base}/fosberg-fire-weather-index.icechunk', consolidated=False)
q99  = xr.open_zarr(f'{base}/fosberg-fire-weather-index_p99.icechunk', consolidated=False)
mode = xr.open_zarr(f'{base}/fosberg-fire-weather-index_p99_mode.icechunk', consolidated=False)
```
