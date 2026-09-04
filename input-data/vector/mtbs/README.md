# MTBS fire burn perimeters

MTBS burned area boundaries (USGS/USFS, national shapefile, 1984-present) ->
`s3://carbonplan-ocr/ocr-explore/mtbs_perims.pmtiles` (layer `mtbs_perims`).
Provenance is merged into `s3://carbonplan-ocr/ocr-explore/manifest.json`.

Not wired into the OCR pipeline CLI — run by hand, with AWS credentials for
carbonplan-ocr. tippecanoe is invoked the same way the pipeline's
`create_*_pmtiles.py` scripts do; the repo's pixi environment already provides
it (`pixi shell`), or use any tippecanoe on PATH:

```
uv run mtbs_perims.py fetch
uv run mtbs_perims.py build --force
uv run mtbs_perims.py verify
```
