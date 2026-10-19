# NOAA Storm Events flood reports

See `../historic-events.md` for how this artifact is used and what it means.

NOAA NCEI Storm Events Database, yearly details CSVs (1996-present), Flood
and Flash Flood events inside CONUS ->
`s3://carbonplan-ocr/ocr-explore/storm_events_conus_floods.pmtiles` (layer
`floods`). Raw CSVs are mirrored under `ocr-explore/StormEvents/raw/`, and
provenance is merged into `s3://carbonplan-ocr/ocr-explore/manifest.json`.

Not wired into the OCR pipeline CLI — run by hand, with AWS credentials for
carbonplan-ocr and tippecanoe on PATH:

```
uv run storm_events_floods.py fetch
uv run storm_events_floods.py build --force
uv run storm_events_floods.py verify --coverage
```

`fetch` takes the newest file per year (NCEI reissues years as reports are
revised; the creation stamp in each file name keeps raw mirrors immutable).
ocr-web reads the reports near a point by decoding max-zoom tiles directly,
so `verify --coverage` decodes every max-zoom tile to confirm no report was
dropped.
