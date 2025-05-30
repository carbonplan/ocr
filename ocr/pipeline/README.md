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

# May 30th thoughts:

We could run coiled batch using the coiled python api
This would give us the state of the job(s)! ex: coiled.batch.status(907924)

```python
batch = coiled.batch.run('pipeline/01_Write_region.py -r y9_x2')

# this coiled.batch.wait is blocking! So it could wait before running more pipeline steps
status = coiled.batch.wait_for_job_done(batch['job_id'])
```

So pipeline design:

1. main.py
   1. accepts click args (region_id vs all vs subset etc.)
   2. Checks icechunk store and inits if needed? We need to do this once, before in parallel?
   3. Uses coiled.batch + status to deploy pipeline:
      1. create template serially (blocking)
      2. For each region_id, get icechunk diff, deploy jobs with coiled batch until every coiled job, cluster? etc. reports done (or failed?) (blocking / empar)
         1. This could be split up later or done in one (ie, 01_Write_Region.py)
      3. geoparquet aggregation (serial, blocking)
         1. aggregate geoparuqet files and write (need batch status)
            1. scale VM based on # of entries in icechunk history
      4. PMTiles creation batch job (can/should we scale based on the # of files in step 3? or icechunk history?).
         1. GDAL wizardry - we could still track status to report pipeline has finished, this might be really nice in a github action.
