## Data pipeline

Ideally we will have 'atomic units' that are based on independent dask chunks. These should be able to be written in parallel using the [Icechunk uncooperative writes strategy](https://icechunk.io/en/latest/icechunk-python/parallel/#uncooperative-distributed-writes).

### Goals

We should be able to:

- Run each atomic unit in an emb-par manner.
- Use coiled batch to deploy units of work.
- Use GH actions to deploy coiled jobs.

### Current flow

() = dataset or intermediate product
[] = calculation / pipeline step
[*] = pipeline step can be run in atomic units

1. (WindSfc 1/4 degree EPSG:4326, USFS community risk 30m EPSG:4326, USFS future risk 30m interpolated EPSG:4326) -> [*Wind Calc] -> (Wind informed risk Icechunk 30m)
2. (Wind informed risk Icechunk 30m) -> [*Building centroid risk selection] -> (wind risk geoparquet by region_id)
3. (wind risk geoparquet by region_id) -> [aggregate geoparquet into CONUS] -> (CONUS geoparuqet)
4. (CONUS geoparuqet) -> [Create PMTiles] -> (PMTiles)
