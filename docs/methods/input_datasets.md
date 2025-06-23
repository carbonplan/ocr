# Input Datasets

Input data provenance can be tracked through the ingestion scripts in `input-data/`. These datasets are added to the `ocr catalog` and can be accessed programmatically.

The catalog **repr** will list available datasets.

```python
from ocr import catalog

catalog
```

## Tensor data

This refers to n-dimensional datasets that fit into the Zarr model.

### USFS Wildfire Risk to Communities

**USFS Wildfire Risk to Communities: Spatial datasets of landscape-wide wildfire risk components for the United States (2nd Edition)**

[data source ](https://www.fs.usda.gov/rds/archive/catalog/RDS-2020-0016-2)

This dataset is distributed at state and CONUS levels for each variable in `.tif` raster files. In `input-data/tensor/USFS_fire_risk/RDS-2020-0016-2` we download and combine these tif's into a single Icechunk store.

```python
from ocr import catalog
rps_30 = catalog.get_dataset(
    'USFS-wildfire-risk-communities'
).to_xarray()
```

### USFS Wildfire Risk 2011 and 2047

**USFS Spatial datasets of probabilistic wildfire risk components for the conterminous United States (270m) for circa 2011 climate and projected future climate circa 2047**

[data source](https://www.fs.usda.gov/rds/archive/catalog/RDS-2020-0016-2)

This dataset is distributed as a single zipped archive containing multiple `.tif` raster files. In `input-data/tensor/USFS_fire_risk/RDS-2025-0006` we download, unzip and combine these tif's into Icechunk stores -- one for each climate run (2011 and 2047).

```python
from ocr import catalog
climate_run_2011 = catalog.get_dataset('2011-climate-run-30m-4326').to_xarray()
climate_run_2047 = catalog.get_dataset('2047-climate-run-30m-4326').to_xarray()
```

### Wind

TODO: Leaving for now since we might switch out the input wind dataset.

## Vector data

In our model we are thinking of vector data as traditional GIS data formats (shapes, lines, polygons etc.)

### Overture buildings dataset

[data source ](https://docs.overturemaps.org/guides/buildings/#14/32.58453/-117.05154/0/60)

In `input-data/vector/overture_vector/` we subset the Overture building geoparuqet into a CONUS extent geoparuqet file.

```python
from ocr import catalog
conus_buildings = catalog.get_dataset('conus-overture-buildings')
```
