## Data pipeline

Ideally we will have 'atomic units' that are based on independent dask chunks. These should be able to be written in parallel.

### Strategy

- We should be able to run each atomic unit in an emb-par manner.
- We can use coiled batch to deploy units of work.
- We can use GH actions to deploy coiled batch jobs.

### Current flow

() = dataset or intermediate product
[] = calculation / pipeline step
[*] = pipeline step can be run in atomic units

1. (WindSfc, USFS 30m risk, USFS 270m climate runs) -> [*Wind Calc] -> (Wind informed risk Icechunk 30m)
2. (Wind informed risk Icechunk 30m) -> [*Building centroid risk selection] -> (wind risk geoparquet by region_id)
3. (wind risk geoparquet by region_id) -> [aggregate geoparquet into CONUS] -> (CONUS geoparuqet)
4. (CONUS geoparuqet) -> [Create PMTiles] -> (PMTiles)
