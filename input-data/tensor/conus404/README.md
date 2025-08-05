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
