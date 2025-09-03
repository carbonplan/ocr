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
│ --help                               Show this message and exit.                                                                                                                      │
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

## Fire-weather modal wind direction

The script `compute_fire_weather_wind_mode.py` computes, for every CONUS404 grid cell, the **most common (modal) wind direction during hours that meet fire weather criteria**. It reproduces and productionizes the notebook logic from `notebooks/fire-weather-wind-mode-reprojected.ipynb`.

### What it does

1. Starts (optionally) a Coiled cluster.
2. Loads required CONUS404 hourly variables (Icechunk-backed) via `load_conus404`.
3. Computes relative humidity from temperature & dewpoint.
4. Rotates grid-relative winds (U10, V10) into earth-relative components using `SINALPHA` / `COSALPHA`.
5. Derives wind speed and direction with `xclim`.
6. Applies a gust factor (default 1.4×) before thresholding wind speed (mph) and relative humidity to build a fire weather mask.
7. Classifies wind direction into 8 cardinal bins (N, NE, E, SE, S, SW, W, NW) for fire-weather hours only.
8. Builds a per-pixel direction histogram and selects the argmax (modal direction) where at least one fire-weather hour exists.
9. Writes a native-grid Zarr.
10. Optionally reprojects to the geobox of the catalog dataset `USFS-wildfire-risk-communities-4326` and writes a second Zarr.

### Cardinal direction encoding

The resulting variable `wind_direction_mode` stores integers 0–7 mapping to:

| Code | Direction |
| ---- | --------- |
| 0    | N         |
| 1    | NE        |
| 2    | E         |
| 3    | SE        |
| 4    | S         |
| 5    | SW        |
| 6    | W         |
| 7    | NW        |

Cells with no hours meeting fire weather criteria are `NaN`.

### CLI help

```bash
pixi run python input-data/tensor/conus404/compute_fire_weather_wind_mode.py --help
```

Expected options (summarized):

| Option                         | Description                                           | Default                                                        |
| ------------------------------ | ----------------------------------------------------- | -------------------------------------------------------------- |
| `--hurs-threshold`             | Relative humidity threshold (%)                       | 15                                                             |
| `--wind-threshold`             | Gust-like wind threshold (mph)                        | 35                                                             |
| `--wind-gust-factor`           | Multiplier applied to sustained speed                 | 1.4                                                            |
| `--output-base`                | Base S3 (or local) path for outputs                   | `s3://carbonplan-ocr/input-data/conus404-wind-direction-modes` |
| `--target-dataset-name`        | Catalog dataset whose geobox is used for reprojection | `USFS-wildfire-risk-communities-4326`                          |
| `--reproject / --no-reproject` | Toggle reprojection                                   | `True`                                                         |
| `--cluster-name`               | Coiled cluster name                                   | `fire-weather-distribution`                                    |
| `--min-workers`                | Min workers (Coiled autoscaling)                      | 4                                                              |
| `--max-workers`                | Max workers                                           | 50                                                             |
| `--worker-vm-types`            | Worker VM type                                        | `m8g.xlarge`                                                   |
| `--scheduler-vm-types`         | Scheduler VM type                                     | `m8g.large`                                                    |
| `--local`                      | Run locally (no Coiled cluster)                       | `False`                                                        |

### Example: run on Coiled (default thresholds)

```bash
pixi run coiled batch run input-data/tensor/conus404/compute_fire_weather_wind_mode.py \
    --hurs-threshold 15 \
    --wind-threshold 35 \
    --wind-gust-factor 1.4
```

### Example: local dry run (small subset)

If you only want to test logic locally (no cluster) you can optionally slice after loading (modify script or use a forked copy):

```bash
pixi run python input-data/tensor/conus404/compute_fire_weather_wind_mode.py --local --reproject False
```

### Outputs

Given thresholds H=15 (RH) and W=35 (wind), two Zarr stores are written by default:

```
s3://carbonplan-ocr/input-data/conus404-wind-direction-modes/
    fire_weather_wind_mode-hurs15_wind35.zarr
    fire_weather_wind_mode-hurs15_wind35-reprojected.zarr
```

The reprojected store matches the grid of the wildfire risk communities dataset. Disable via `--no-reproject`.

### Reading the output

```python
import xarray as xr

path = 's3://carbonplan-ocr/input-data/conus404-wind-direction-modes/fire_weather_wind_mode-hurs15_wind35.zarr'
mode = xr.open_zarr(path)
mode.wind_direction_mode
```

### Adjusting thresholds

Simply change `--hurs-threshold`, `--wind-threshold`, and (optionally) `--wind-gust-factor`. The output filenames embed these values so multiple experimental runs can co-exist in the same prefix.

### Notes / Future Enhancements

- Add optional temperature threshold when/if needed (currently disabled like the notebook).
- Option to emit the full directional histogram, not just the mode.
- Potential integration tests on a spatial subset.
