# IBTrACS hurricane tracks

NOAA NCEI IBTrACS v04r01 since-1980 lines shapefile ->
`s3://carbonplan-ocr/ocr-explore/ibtracs_since1980.pmtiles` (layer `tracks`),
read by the ocr-explore viewer (`datasets/ibtracs.tsx`). Provenance is merged
into `s3://carbonplan-ocr/ocr-explore/manifest.json`.

Not wired into the OCR pipeline CLI — run by hand, with AWS credentials for
carbonplan-ocr. tippecanoe is invoked the same way the pipeline's
`create_*_pmtiles.py` scripts do; the repo's pixi environment already provides
it (`pixi shell`), or use any tippecanoe on PATH:

```
uv run ibtracs_tracks.py fetch
uv run ibtracs_tracks.py build --force
uv run ibtracs_tracks.py verify
```

## CONUS wind footprints (ocr-web "Previous storms")

`ibtracs_winds.py` runs every since-1980 storm that carries tropical-storm
winds within 5 degrees of CONUS through the CHAZ hazard maps' wind step
(`chaz_events.winds_parallel` from `input-data/tensor/chaz/climada`: CLIMADA
Holland 2008 on the CHAZ 300 arcsec centroids, 30 min resampling) and packs
the peak wind per storm per cell onto the served CHAZ grid as
`s3://carbonplan-ocr/ocr-explore/ibtracs_conus_winds.zarr`
(`wind(storm, lat, lon)`, uint8 m/s, storm table in the group attrs).

Before the wind step each track's inputs are normalized the way CLIMADA's
CHAZ reader prepares synthetic tracks: central pressure estimated from wind
and position, environmental pressure fixed at 1010 hPa, radius of maximum
wind estimated from pressure. A historic storm is then exactly what the hazard
model produces for that track and intensity, so its wind at a cell can be read
against the cell's return-period curve. `--observed-inputs` keeps NHC's
pressure and radius of maximum wind for validation runs only.

The `winds` stage needs the `climada` pixi environment and `chaz_events.py` on
the import path; `store` runs in the default one. `build --subset conus` then
tiles only those storms' tracks to `ibtracs_conus_tracks.pmtiles`, so the
drawn tracks and the footprints agree:

```
pixi run -e climada python ibtracs_winds.py winds
pixi run python ibtracs_winds.py store --force
uv run ibtracs_tracks.py build --subset conus --force
uv run ibtracs_tracks.py verify --subset conus
```
