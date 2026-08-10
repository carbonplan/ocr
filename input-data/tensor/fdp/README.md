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

## Outputs

| level | resolution | shape       |
| ----- | ---------- | ----------- |
| 0     | 100 m      | 31327×48357 |
| 7     | 12.8 km    | 244×377     |

```
ocr-explore/FDP/raw/CONUS_FDP_100m.tif
ocr-explore/FDP/processed/flood_damage_probability/{0..7}/
ocr-explore/FDP/processed/manifest.json
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
  presence/absence raster derived from a threshold on this surface. The viewer
  reproduces that with a client-side `threshold` uniform instead of hosting the
  second raster.
