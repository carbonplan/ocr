# United States Flood Database

See `../historic-events.md` (Flood source comparison) for how this is used.

USFD v1.1 (Li et al. 2021, ESSD,
[doi:10.5194/essd-13-3755-2021](https://doi.org/10.5194/essd-13-3755-2021);
data [doi:10.5281/zenodo.7545697](https://doi.org/10.5281/zenodo.7545697),
CC BY 4.0) merges seven flood databases into one CSV. Rows with coordinates
in the CONUS bounding box ->
`s3://carbonplan-ocr/ocr-explore/usfd_conus.pmtiles` (layer `usfd`), one
point each. The raw CSV is mirrored at `ocr-explore/USFD/raw/`, and
provenance is merged into `ocr-explore/manifest.json`.

v1.0 stores most dates in scientific notation (`2.01E+13`), so only v1.1 is
usable. Row ids repeat in the source, so features carry their row number as
`id`.

Run by hand, with AWS credentials for carbonplan-ocr and tippecanoe on PATH:

```
uv run usfd_floods.py fetch
uv run usfd_floods.py build --force
uv run usfd_floods.py verify --coverage
```
