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
