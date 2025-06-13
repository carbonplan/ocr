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

### Current layout:

1. main.py
   1. accepts click args (region_id(s) vs all, logging etc.)
   2. Checks icechunk store and inits with template if needed. (TODO: Add option to overwrite)
   3. Uses coiled.batch + status to deploy pipeline: 2. For each region_id, get icechunk diff (TODO), deploy jobs with coiled batch. TODO: Could use monitoring with coiled batch for blocking.
      1. This could be split up later or done in one. Currently in `01_write_region.py`
      2. geoparquet aggregation (serial, blocking)
         1. aggregate geoparuqet files and write (TODO: need batch status for blocking)
            1. scale VM based on # of entries in icechunk history? (TODO)
            2. TODO: We could potentially switch methods to gpq.
      3. PMTiles creation batch job (can/should we scale based on the # of files in step 3? or icechunk history? TODO).
         1. GDAL wizardry - we could still track status to report pipeline has finished, this might be really nice in a github action.
