# MTBS fire burn perimeters

MTBS burned area boundaries (USGS/USFS, national shapefile, 1984-present) ->
`s3://carbonplan-ocr/ocr-explore/mtbs_perims.pmtiles` (layer `mtbs_perims`).
Provenance is merged into `s3://carbonplan-ocr/ocr-explore/manifest.json`.

Not wired into the OCR pipeline — run by hand (needs tippecanoe on PATH and
AWS credentials for carbonplan-ocr):

```
uv run mtbs_perims.py fetch
uv run mtbs_perims.py build --force
uv run mtbs_perims.py verify
```
