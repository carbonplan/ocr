# USGS Flood Damage Probability (CONUS)

- Data: https://doi.org/10.5066/P954TTQN (USGS ScienceBase, public domain)
- Paper: Collins et al. (2022), _Environmental Research Letters_ 17, 034006.
  https://doi.org/10.1088/1748-9326/ac4f0f

Random-forest-predicted probability of flood damage, 0..1, on a 100 m
31327 × 48357 CONUS grid in NAD83 / Conus Albers (EPSG:5070).

Like the CHAZ scripts, this is run by hand and writes to the `ocr-explore`
prefix, where the exploration viewer reads it. It is not wired into the OCR
pipeline.

For the proposed temporal validation and retraining with newer NOAA reports,
see [the validation and rerun specification](RERUN_SPEC.md). The first commands
below package the published raster; [Retraining](#retraining) refits the model.

The executed [local validation notebook](validation-2026-10-15/validation.ipynb)
compares the frozen raster with June 2020–December 2025 NOAA reports. An
[HTML version](validation-2026-10-15/validation.html) and a
[run record](validation-2026-10-15/README.md) accompany the results, limitations,
input hashes and reproducible local validation script.

## Running it

`build` moves ~12 GB and holds a 6 GB array in memory, so run it next to the
bucket. Credentials forward automatically and the script has no Coiled
dependency, so it runs anywhere with enough RAM.

Use the `fdp` pixi environment: Coiled package sync replicates whatever
environment it is launched from, and the default one carries packages its solver
cannot resolve (`pydantic-extra-types`).

```bash
pixi run -e fdp coiled run --region us-west-2 --vm-type m8g.2xlarge --disk-size 60 -- python input-data/tensor/fdp/fdp.py build

pixi run -e fdp python input-data/tensor/fdp/fdp.py verify
```

`coiled run` uploads any argument that is a local file and rewrites the path, so
the script needs no `--file`.

`build` resolves the source raster, builds the pyramid, syncs it, writes a
`manifest.json` record alongside, and verifies what it wrote. The source comes
from the local cache, else the raw copy in S3, else a 3 GB zip from ScienceBase
that it stages under `FDP/raw/` on the way through, so the slow download happens
once ever. Budget ~9 GB of cache disk (`~/.cache/ocr/fdp`, `FDP_CACHE`
overrides); the zip is left behind once the tif is extracted.

`verify` checks levels present and float32, shapes halving, values in [0, 1],
`proj:code`, and the manifest record. Non-zero exit on any violation.

The sync deletes keys under the store prefix that the build did not write, so a
change in shape or chunking leaves no orphaned chunks behind.

## Building footprints

`fdp_buildings.py` samples the store onto the ~156M Overture CONUS footprints
from the fire pipeline's region-tagged buildings source, writing one GeoParquet
with an `fdp` column.

```bash
pixi run -e fdp coiled run --region us-west-2 --vm-type m8g.4xlarge --disk-size 200 -- python input-data/tensor/fdp/fdp_buildings.py build

pixi run -e fdp python input-data/tensor/fdp/fdp_buildings.py verify
```

Each building takes the value of the cell nearest its bbox centroid. At 100 m
that is close to footprint scale, so these read as per-building values rather
than the cell expectations a coarse grid gives; buildings larger than a cell
still get one value, and neighbours inside the same cell share one.

Footprints are in lon/lat and the grid is in Albers metres, so centroids are
reprojected before the cell arithmetic. The full grid is 1.5e9 cells, too many
to join against, so a first pass collects the cells that actually contain a
building and only those become the join table. That pass reads just the `bbox`
column, which parquet column pruning makes much cheaper than the full scan.
Buildings outside the store's valid data are dropped by the join, so every
written row has a value.

Rows are sorted into grid row-major order so bbox-filtered reads prune row
groups. The sort spills, hence the disk headroom above.

## Retraining

Refits the paper's 18 HUC-2 random forests with `train_fdp.R`, a port of the
released R script (see its header for what differs). Three steps, each
resumable and each writing under `ocr-explore/FDP/`:

```bash
# 1. predictors, released points and the HUC-2 raster -> FDP/inputs/ (~18 GB of downloads)
pixi run -e fdp python input-data/tensor/fdp/fdp_inputs.py stage

# 2. training points for each run -> FDP/runs/<run>/points.csv
pixi run -e fdp python input-data/tensor/fdp/fdp_points.py build

# 3. one watershed here, or all of them on Coiled
pixi run -e fdp-r python input-data/tensor/fdp/fdp_train.py task --run original --huc 03
pixi run -e fdp python input-data/tensor/fdp/fdp_train.py launch --run noaa-200612-202512
```

Runs:

- `original`: the released points, to check that the port reproduces the
  published surface.
- `noaa-200612-202512`: every Flood/Flash Flood report from Dec 2006 through
  Dec 2025 at its begin coordinates, with pseudo-absences resampled by the
  paper's rules. `points.json` records counts, the NOAA files and their hashes.
- `noaa-200612-202005`: the same construction over the released points'
  window. Against `original` it isolates the effect of rebuilding the points;
  against `noaa-200612-202512`, the effect of the added years.

`--distance planar` corrects the spatial-sorting distances that the released
script computes as if Albers metres were degrees (see
[the rerun specification](RERUN_SPEC.md#data-readiness)); the default keeps the
original behaviour. `--quick` shrinks tuning and the forest for a smoke test.

Locally, the `fdp-r` environment needs four CRAN packages that conda-forge does
not build for macOS. Compile them once:

```bash
pixi run -e fdp-r fdp-r-cran
```

On Coiled, `launch` adds the R packages to the synced `fdp` environment as conda
extras, one x86 VM per watershed (dismo has no ARM build). The paper reports an
average of 5.55 h per watershed on 8 cores with ~23 GB RAM; each task's
`run.json` records its actual time, peak memory and disk.

A task's cut of the predictors is cached per watershed under
`~/.cache/ocr/fdp/predictors/` (1.6 GB for HUC 03) and reused while the staged
inputs are unchanged, so repeat runs of a watershed skip the ~8 min of reads.

Each task uploads `fdp.tif` (the watershed's probabilities), `summary.json`
(CV AUCs, thresholds, tuned parameters, OOB Brier, stage timings),
`compare.json` (agreement with the published surface over the same cells, and
separately outside the 100-yr floodplain), `importance.csv`, the final ranger
model, CV predictions and the R log.

The 100-yr floodplain holds no pseudo-absences, so the model scores it near 1
throughout: in HUC 03 every floodplain cell (20% of the watershed) falls in the
top fifth of the surface. Agreement over the whole watershed is therefore
inflated by a block both surfaces get trivially right, and the
outside-floodplain figures are the informative ones.

### Driver attributions

`task --contributions` also runs `contrib_fdp.R`, which splits every cell's
probability into one contribution per predictor along its tree paths (Saabas
attribution, compiled from `fdp_contrib.cpp`): per cell, baseline +
contributions = probability. It writes `contributions.tif` (int16 bands,
x 1e-4) and `contributions.json` (driver order, baseline, checks). Two checks
fail the task: the internal node values must reproduce the leaf values ranger
stored, and the reconstructed probabilities must match `fdp.tif` (to 3e-8 on
HUC 06). It adds about the time of the prediction (~20 min for HUC 03).

The attributions explain the model, not causation: correlated terrain
predictors share credit arbitrarily, and inside the floodplain nearly all of
the probability is credited to the floodplain, for the reason above.

`fdp_retrained.py` mosaics a run's watersheds onto the published grid as a
served store with `fdp`, `contribution (driver, y, x)` and `dominant_driver`:

```bash
pixi run -e fdp python input-data/tensor/fdp/fdp_retrained.py build --run noaa-200612-202512 --distance planar --hucs 03
```

The store is cropped to the watersheds included and holds the whole level in
memory, so all 18 at once need a large VM (contributions alone are ~33 GB as
int16 at CONUS scale). Coarse levels average `fdp` and `contribution`, so the
sum identity holds at every level up to int16 truncation (2.4e-3 at level 7);
`dominant_driver` is recomputed from each level's contributions. `verify`
checks both. The ocr-explore layer _Flood Damage Probability Drivers_ reads it.

## Outputs

| level | resolution | shape       |
| ----- | ---------- | ----------- |
| 0     | 100 m      | 31327×48357 |
| 7     | 12.8 km    | 244×377     |

```
ocr-explore/FDP/raw/CONUS_FDP_100m.tif
ocr-explore/FDP/processed/flood_damage_probability/{0..7}/
ocr-explore/FDP/processed/manifest.json
ocr-explore/FDP/buildings/fdp_buildings_conus.parquet
ocr-explore/FDP/buildings/manifest.json
ocr-explore/FDP/inputs/predictors/{floodplain,distwater,...}.tif
ocr-explore/FDP/inputs/response/{presence,absence}.csv
ocr-explore/FDP/inputs/huc2_100m.tif
ocr-explore/FDP/inputs/manifest.json
ocr-explore/FDP/runs/<run>/points.{csv,json}
ocr-explore/FDP/processed/retrained/<run>-<distance>/{0..7}/
ocr-explore/FDP/runs/<run>/<distance>[-quick]/huc<NN>/
```

Level `0` is the native grid; `x`/`y` are Albers metres, not degrees.

```python
import xarray as xr

base = 's3://carbonplan-ocr/ocr-explore/FDP/processed'
ds = xr.open_zarr(f'{base}/flood_damage_probability/0', consolidated=False)
```

## Caveats

- **Probability of damage, not depth or loss.** The response variable is NOAA
  Storm Events flood-damage reports (presence) against sampled absence points,
  so the layer ranks where damage has been reported and is likely, not how much.
  It carries no depth, frequency, or return-period information.
- **Trained on reported damage.** Presence points inherit the reporting biases
  of the NOAA record: populated, well-observed places are better represented.
- **Historical.** The predictors (2016 land cover, road density, floodplain,
  terrain) describe recent conditions; there is no climate scenario dimension.
- **Continuous vs. binary.** The published product also ships a binary
  presence/absence raster, `Output_CONUS_binary_pres_abs_100m`. Its cut is not
  one number: `Code_FDP_RandomForest_code.R` fits a separate error-minimizer
  threshold per HUC-2 watershed, the probability where that watershed's false
  positive and false negative rates cross, and the release ships the code but
  not the resulting values. No client-side threshold reproduces it, so the
  viewer carries the continuous surface alone.
