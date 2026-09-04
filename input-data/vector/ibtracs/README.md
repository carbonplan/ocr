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
winds within 5 degrees of CONUS through CLIMADA's Holland (2008) wind model on
the CHAZ 300 arcsec centroids, and packs the peak wind per storm per cell onto
the served CHAZ grid as `s3://carbonplan-ocr/ocr-explore/ibtracs_conus_winds.zarr`
(`wind(storm, lat, lon)`, uint8 m/s, storm table in the group attrs). The
`winds` stage needs the `climada` pixi environment; `store` runs in the default
one. `build --subset conus` then tiles only those storms' tracks to
`ibtracs_conus_tracks.pmtiles`, so the drawn tracks and the footprints agree:

```
pixi run -e climada python ibtracs_winds.py winds
pixi run python ibtracs_winds.py store --force
uv run ibtracs_tracks.py build --subset conus --force
uv run ibtracs_tracks.py verify --subset conus
```
