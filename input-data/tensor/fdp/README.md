# USGS Flood Damage Probability (CONUS)

- Data: https://doi.org/10.5066/P954TTQN (USGS ScienceBase, public domain)
- Paper: Collins et al. (2022), _Environmental Research Letters_ 17, 034006.
  https://doi.org/10.1088/1748-9326/ac4f0f

Random-forest-predicted probability of flood damage, 0..1, on a 100 m
31327 × 48357 CONUS grid in NAD83 / Conus Albers (EPSG:5070).

Like the CHAZ scripts, this is run by hand and writes to the `ocr-explore`
prefix, where the exploration viewer reads it. It is not wired into the OCR
pipeline.

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
