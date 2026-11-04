# Data pipeline

Each risk runs as an ordered list of stages: `etl`, `score`, `pyramid`, `buildings`, `regions`. Storage is the hand-off: each stage reads what an earlier one wrote. `pyramid` and `buildings` each read only the risk store (`store.icechunk`); `regions` reads the buildings file. A stage runs in process, or as one Coiled batch job with `--coiled`.

Fire still uses the older `ocr-deploy` CLI (`ocr-deploy --help`) until its refactor.

## Run a deployment

A deployment is one committed TOML file under `runs/<risk>/`:

```bash
pixi run -e flood flood-etl --version v2.0.0  # once: writes store.icechunk (add --overwrite to redo)
ocr run runs/flood/v2.0.0.toml --dry-run      # print the plan, check inputs, run nothing
ocr run runs/flood/v2.0.0.toml                # or: pixi run flood-all
```

```toml
risk = "flood"
version = "v2.0.0"
stages = ["pyramid", "buildings", "regions"]  # etl runs separately (flood-etl)
coiled = true

[kwargs.buildings]          # keyword args of the stage's utils, checked before any job starts
point_strategy = "ST_Centroid"

[job.buildings]             # optional coiled.batch.run overrides, over `ocr.cli.JOBS`
vm_type = ["r7g.8xlarge"]
```

## Stage kwargs

`[kwargs.<stage>]` in a run file, or `--kwargs '{"<stage>": {...}}'`, sets keyword args of the stage's utils (`STAGE_KWARGS` in `ocr/stages.py`): pyramid `write_multiscales`, `create_pyramid`; buildings `write_buildings`, `write_buildings_lod`; regions `write_regions`. Names and types are checked before any job starts; a bad name fails and the error lists the valid ones. `--kwargs` on `ocr run` merges key by key over the file's `[kwargs.<stage>]`.

| kwarg | default | meaning |
| --- | --- | --- |
| `pyramid.levels` | from `create_pyramid` | number of levels, or `"auto"` |
| `pyramid.branch` | `"main"` | icechunk branch to write |
| `buildings.bbox` | CONUS | lon/lat subset (see below) |
| `buildings.point_strategy` | `"ST_Centroid"` | or `"ST_PointOnSurface"` |
| `buildings.memory_limit`, `.workers` | DuckDB's | DuckDB limits for the LOD step |
| `regions.levels` | all | e.g. `["states", "counties"]` |

`[job.<stage>]` takes any `JobConfig` field (`ocr/cli.py`), e.g. `vm_type`, `disk_size`, `env`, `tag`, `spot_policy`, `job_timeout`, over the defaults in `ocr.cli.JOBS`.

## Rerun some stages

Rerun with the same file's settings, using `--stages` and `--kwargs`:

```bash
ocr run runs/flood/v2.0.0.toml --stages regions --kwargs '{"regions": {"levels": ["counties"]}}'
```

Or name everything on the command line:

```bash
ocr stage --version v2.0.0 --coiled flood pyramid
```

`ocr run --help` and `ocr stage --help` list more examples.

- Stages always run in pipeline order, whatever order you give them.
- A failed stage stops the run; later stages don't start. Rerun from it with `--stages`.
- A stage you skip must already have its input in storage, or the run fails before it starts. So must TIGER (`pixi run regions-etl`) for `buildings` and `regions`.
- A stage the risk has no code for also fails before any job starts.
- Flood `etl` on an existing store fails before any job starts; pass `--overwrite` or leave `etl` out of `--stages`.
- `--overwrite` deletes a stage's old output first, where that applies (`score`, `pyramid`, `regions`). `buildings` and `regions` replace their files on every run; pass `--overwrite` to `regions` after a smaller run (`bbox`, fewer `levels`) so no old files stay.
- Coiled jobs print their job and cluster id and a `coiled batch logs <cluster id>` command when submitted.
- `--vm TYPE` sets the Coiled VM type for every stage in the run.

## Storage and environment

| variable | meaning |
| --- | --- |
| `OCR_STORAGE_BUCKET`, `OCR_STORAGE_REGION` | S3 bucket and region. Forwarded to Coiled jobs. |
| `OCR_STORAGE_LOCAL_ROOT` | Write under a local directory instead of S3. Not allowed with `--coiled`. |

A run file can set `bucket = "..."` (wins over `OCR_STORAGE_BUCKET`) and `prefix = "..."`, the output prefix (default `output`; inputs stay under `input/`). The prefix has no env var: it comes from the run file or `ocr stage --prefix`, and Coiled jobs get it as `--prefix`.

Scratch runs: use a prerelease version (`v2.0.0-scratch`, `v0.0.0-nrh`), or a scratch `prefix`. The version must be semver with a leading `v`.

Local smoke run: `OCR_STORAGE_LOCAL_ROOT=~/ocr-data pixi run flood-local` (`runs/flood/local.toml`).

Subset: `bbox` (lon/lat xmin, ymin, xmax, ymax) on `buildings` keeps only buildings inside it, and `regions` then covers only those, with partial stats for regions cut by the box. So a subset never replaces production outputs, `bbox` needs a scratch `prefix` or a prerelease version (the run fails before it starts otherwise):

```bash
ocr stage --version v2.0.0-dev --kwargs '{"buildings": {"bbox": [-122.5, 37.7, -122.3, 37.8]}}' flood buildings
```

## How it works

`ocr run` and `ocr stage` (`ocr/cli.py`) run a risk's stages one at a time, in `STAGES` order (`ocr/stages.py`). There is no graph and no scheduler, only a for-loop in `execute()`. Paths are under `<prefix>/<risk>/v<version>/` (prefix default `output`) (`OutputPaths.store`, `.buildings`, `.regions`).

| stage | util | reads (`NEEDS`) | writes |
| --- | --- | --- | --- |
| etl | `RiskSpec.etl` | raw source | flood: `store.icechunk` group "0"; others: `input/<risk>/...` |
| score (fire only) | `RiskSpec.score` | ETL output (not checked) | `store.icechunk` group "0" |
| pyramid | `write_multiscales` | `store.icechunk` | `store.icechunk` groups 1..N |
| buildings | `write_buildings` | `store.icechunk` | `vector/buildings.geoparquet` |
| regions | `write_regions` | `vector/buildings.geoparquet` | `vector/regions/` |

The store is written by `score`, or by `etl` when the risk has no score (flood). Before anything runs, `check_storage()` and `unimplemented()` do the preflight checks listed under [Rerun some stages](#rerun-some-stages), so Coiled never boots a VM just to raise.

Coiled job settings stack in this order, last wins: `DEFAULT`, then runtime (bucket region, forwarded storage env), then `JOBS[stage]`, then the run file's `[job.<stage>]`, then `--vm`. `env` and `tag` merge key by key. Each job runs `ocr stage --version V <risk> <stage> [--overwrite]`, because the VM doesn't have the run file. Its kwargs go as JSON in the `OCR_STAGE_KWARGS` env var, not as `--kwargs`: Coiled rebuilds the command as a shell string and strips the JSON quotes.

## Change the pipeline

| change | where |
| --- | --- |
| new stage | A function `(spec, out, overwrite, **kw)` and a `STAGES` entry at its place in run order. Add `NEEDS` if it reads another stage's output (and, for a new output, its writer in `check_storage`'s `writers`), `STAGE_KWARGS` if it takes kwargs, an `OutputPaths` property for a new output, and a `cli.JOBS` default. |
| stage only some risks have | A `RiskSpec` field (None means no code) and an entry in `unimplemented()`. |
| new risk | `Risk` (`ocr/config.py`), a `RISKS` entry, and `runs/<risk>/vX.Y.Z.toml`. |
| new kwarg on a util | Nothing: `stage_kwargs()` reads the signature (needs a default and a type hint). If the util forwards `**kwargs`, add the receiving util to its `STAGE_KWARGS` tuple. |

`test_run_files_dry_run` dry-runs every `runs/**/*.toml`. After a change, run:

```bash
pixi run -e dev pytest tests/test_cli.py tests/test_stages.py tests/flood/test_flood_etl.py
```
