# Global Flood Database

See `../historic-events.md` (Flood source comparison) for how this is used.

Global Flood Database v1 (Tellman et al. 2021, Nature,
[doi:10.1038/s41586-021-03695-w](https://doi.org/10.1038/s41586-021-03695-w),
CC BY-NC 4.0): 913 Dartmouth Flood Observatory events from 2000 to 2018,
mapped from MODIS at about 250 m. Cloud to Street publishes each event as a
GeoTIFF in the public `gs://gfd_v3` bucket (large events are split into
several tiles), and the event table as
`data/gfd_qcdatabase_2019_08_01.csv` in
[cloudtostreet/MODIS_GlobalFloodDatabase](https://github.com/cloudtostreet/MODIS_GlobalFloodDatabase).
The same data is `GLOBAL_FLOOD_DB/MODIS_EVENTS/V1` in Earth Engine, which
needs an authenticated account.

The 114 tiles overlapping the CONUS bounding box hold 107 events that flood
a cell inside it (2001-2018). A cell is flooded for an event when `flooded`
is 1 and `jrc_perm_water` is not. Outputs:

- `s3://carbonplan-ocr/ocr-explore/gfd_conus_footprints.pmtiles` (layer
  `gfd`): one Polygon per connected flooded patch with its event's fields.
- `ocr-explore/GFD/gfd_conus_flood_count` and
  `ocr-explore/GFD/gfd_conus_last_flood_year`: topozarr pyramids on the
  native grid, NaN where no event flooded, coarsened by max.

Raw tiles and the event table are mirrored at `ocr-explore/GFD/raw/`, and
provenance is merged into `ocr-explore/manifest.json`. The license is
non-commercial.

Run by hand, with AWS credentials for carbonplan-ocr and tippecanoe on PATH:

```
uv run gfd_floods.py fetch
uv run gfd_floods.py build --force
uv run gfd_floods.py verify --coverage
```
