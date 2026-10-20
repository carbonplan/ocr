# Groundsource flood records

See `../historic-events.md` (Flood source comparison) for how these are used.

Google Research Groundsource (2026, CC BY 4.0,
[doi:10.5281/zenodo.18647054](https://doi.org/10.5281/zenodo.18647054)):
flood records Gemini extracted from news articles, one GeoParquet file of 2.6M
records worldwide. Each record is a uuid, a geocoded place polygon, its area,
and start and end dates; there is no article text, source, or damage. The
authors estimate about 75% precision.

Records whose polygon centroid falls in the CONUS bounding box (so including
border areas of Canada and Mexico) ->
`s3://carbonplan-ocr/ocr-explore/groundsource_conus_footprints.pmtiles` and
`groundsource_conus_centroids.pmtiles` (layer `groundsource`). The raw
parquet is mirrored at `ocr-explore/Groundsource/raw/`, and provenance is
merged into `ocr-explore/manifest.json`.

Run by hand, with AWS credentials for carbonplan-ocr and tippecanoe on PATH:

```
uv run groundsource_floods.py fetch
uv run groundsource_floods.py build --artifact footprints --force
uv run groundsource_floods.py build --artifact centroids --force
uv run groundsource_floods.py verify --artifact centroids --coverage
```

`verify --coverage` on footprints reports one record missing at z12.
