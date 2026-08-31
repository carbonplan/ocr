# IBTrACS hurricane tracks

NOAA NCEI IBTrACS v04r01 since-1980 lines shapefile ->
`s3://carbonplan-ocr/ocr-explore/ibtracs_since1980.pmtiles` (layer `tracks`),
read by the ocr-explore viewer (`datasets/ibtracs.tsx`). Provenance is merged
into `s3://carbonplan-ocr/ocr-explore/manifest.json`.

Not wired into the OCR pipeline — run by hand (needs tippecanoe on PATH and
AWS credentials for carbonplan-ocr):

```
uv run ibtracs_tracks.py fetch
uv run ibtracs_tracks.py build --force
uv run ibtracs_tracks.py verify
```
