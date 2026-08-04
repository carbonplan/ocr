# Global coastal wind hazard maps from the CHAZ tropical cyclone model

- Data: https://datadryad.org/dataset/doi:10.5061/dryad.qfttdz0vz (CC0)
- Paper: Meiler et al. (2026), _Scientific Data_ 13, 136 —
  https://doi.org/10.1038/s41597-025-06452-0
- Upstream code: https://github.com/simonameiler/CHAZ-hazard-maps

Exploratory processing for a tropical-cyclone hazard layer. These scripts are
not wired into the OCR pipeline — they are run by hand and write to the
`ocr-explore` prefix, where the exploration viewer reads them.

The three scripts run in order:

1. `fetch_chaz.py` — stream the raw Dryad files into S3.
2. `chaz_matrix.py` — build the GCM-stacked wind-hazard pyramid matrix.
3. `chaz_damage.py` — apply a TC impact function to get CONUS damage fractions.

`chaz_damage.py` imports `chaz_matrix.py`, so run both from this directory (or
by path, as below — the script's own directory is on `sys.path`). Set up the
environment with `pixi` (see the contribution guide).

## 1. Fetching the raw data

Dryad gates downloads behind a 10-hour bearer token, but the download endpoint
302s to a presigned S3 URL that supports range requests, so the multi-GB zips
are never downloaded: the script opens each one over HTTPS with `fsspec` and
copies members one at a time into S3. Members already present at the same size
are skipped, so interrupted runs resume.

Grab a token from your Dryad account page, then:

```bash
export DRYAD_TOKEN=<paste 10h token>

pixi run python input-data/tensor/chaz/fetch_chaz.py list
pixi run python input-data/tensor/chaz/fetch_chaz.py fetch return_periods --only '*raster.nc' --dry-run
pixi run python input-data/tensor/chaz/fetch_chaz.py fetch return_periods --only '*raster.nc'
pixi run python input-data/tensor/chaz/fetch_chaz.py fetch exceedance_intensity
```

Files land under `s3://carbonplan-ocr/ocr-explore/CHAZ/<product>/`, with the
leading `<product>/` dir inside the zip stripped and macOS cruft dropped.

## 2. Building the hazard matrix

`chaz_matrix.py` writes one multi-band store per
`(metric, scenario, period, variant)`, with dims `(gcm, lat, lon)` and every
band as a variable. The viewer picks a GCM by selector index and a band by
variable name.

|          |                                                                                                                    |
| -------- | ------------------------------------------------------------------------------------------------------------------ |
| origins  | ERA5 + 6 GCMs (CESM2, CNRM-CM6-1, EC-Earth3, IPSL-CM6A-LR, MIROC6, UKESM1-0-LL)                                    |
| scenario | `ssp245` `ssp370` `ssp585`                                                                                         |
| period   | `base` `fut1` `fut2`                                                                                               |
| variant  | `CRH` `SD` (genesis)                                                                                               |
| bands    | `return_periods`: `thr_33` `thr_50`<br>`exceedance_intensity`: `rp_10` `rp_25` `rp_50` `rp_100` `rp_250` `rp_1000` |

That is 18 GCM-stacked stores + 18 median stores + 1 ERA5 = 37 stores per
metric. Missing GCMs are NaN-padded rather than compacted, so a model's index
on the `gcm` axis never shifts. Each combination also gets a precomputed,
NaN-aware median store, so median map and query values agree.

`--representation points` (the default) scatters the native point files onto
the 1/12 degree grid with no over-ocean interpolation; `raster` reads the
published gridded product (a 180 arcsec interpolation of the points) instead.
The points are the model's computation centroids — 300 arcsec over land, 3600
arcsec over ocean — so the grid is inferred from the _modal_ step and phase of
the coordinates; anchoring at the raw minimum would collide ~41% of the ERA5
points into NaN bands.

```bash
pixi run python input-data/tensor/chaz/chaz_matrix.py list
pixi run python input-data/tensor/chaz/chaz_matrix.py build --metric return_periods --scenario ssp245 --period fut1
pixi run python input-data/tensor/chaz/chaz_matrix.py build --all
pixi run python input-data/tensor/chaz/chaz_matrix.py verify chaz_exceedance_intensity_ssp245_fut1_CRH_points
```

Stores are written to `s3://carbonplan-ocr/ocr-explore/CHAZ/processed/<id>/`
and registered in a shared `manifest.json` alongside them.

## 3. Damage fraction and expected annual damage (CONUS)

`chaz_damage.py` derives damage layers from the wind stores by applying the
regionally calibrated TC impact function of
[Eberenz et al. 2021](https://doi.org/10.5194/nhess-21-393-2021) — the same
functions shipped in CLIMADA.

### What it does

**Damage fraction at each return period.** The impact function is the
Emanuel (2011) sigmoid,

```
vn  = max(V − v_thresh, 0) / (v_half − v_thresh)
D   = vn³ / (1 + vn³)          damage fraction, 0..1
```

with `v_thresh = 25.7 m/s` held fixed and `v_half` calibrated per world region
against ~470 historical events. CONUS falls in region NA2 (USA & Canada):
`v_half` = 89.2 m/s for the paper's TDR calibration (CLIMADA's default), 86.0
for RMSF; the TDR1.5 variant (80.5) ships with CLIMADA's calibration files but
is not in the paper. `check-calibration` re-derives the vendored values from
the CSVs in the CLIMADA repository. The function is evaluated the way CLIMADA
evaluates it — sampled every 5 m/s and linearly interpolated, not in closed
form (a subtlety pointed out by S. Meiler; the closed form reads ~5% low on
CONUS EAD). Because damage is monotonic in wind, each `rp_*` band maps
through the function directly. Per-GCM stores are transformed
member-by-member; median stores take the NaN-aware median of the transformed
members. The maps' wind convention (parametric winds, no terrain-roughness
downscaling) matches the calibration's, so the raw maps are the right input.

**Expected annual damage (`ead`, yr⁻¹).** The integral of damage over annual
exceedance frequency, reconstructed from the six return levels: wind is
piecewise-linear in log T between the bands, extended outside them on the end
slopes (down to T = 1, out to T ≈ 32,000), capped at `WIND_CAP` = 90 m/s, and
evaluated through the impact function on a dense log-T grid. The cap bounds
only the extrapolation — it clears the largest measured wind in any store
(81.9 m/s) and sits near the ~85 m/s observational record; walkthrough §5
scores moving it. Against the event-set EAD Simona Meiler computed from the
full ERA5 set, the integral lands within 1.6% in aggregate on 44,029 shared
US cells and within 1% in the cells carrying two thirds of the loss; the
residual is the tail extrapolation, and walkthrough Part 4 works through it,
including why the alternatives (truncating or holding damage flat past
`rp_1000`) score worse. The convention is recorded per store as
`ead_convention`.

**Vulnerability envelope (`ead_lower` / `ead_upper`).** The same integral
with `v_half` at the 75th/25th percentiles of the per-event calibration fits
(NA2: 74.05–115.1 m/s) — the dominant uncertainty for absolute TC risk
(Meiler et al. 2025). The envelope is roughly a factor of two each way
(Miami: 0.09% / 0.24% / 0.50% per yr).

**NA2 mask.** The NA2 function is the only one applied, so the grid is
clipped to NA2 land. Cells the CONUS bbox reaches that belong to other
regions (Mexican coastline, northern Bahamas — both NA1 in the paper) are
masked rather than given the wrong function, as is open water; walkthrough
§13 prices the wrong-function alternative at roughly 4×. The mask rasterizes
vendored Natural Earth outlines (`na2_geometry.npz`, pinned to v5.1.2) with
`all_touched=True`, so a cell survives if any part of it is NA2 and no US or
Canadian cell is dropped. Per-grid masks are cached in `na2_mask.npz`;
`fetch-geometry` re-vendors the outlines for a new Natural Earth release.

`ead` is the wind analog of the fire product's `RPS = BP × cRPS`: an expected
annual damage fraction for a generic structure — Miami ~0.24%/yr, CONUS max
~1%/yr, the same range as OCR's fire risk-of-loss.

### Commands

```bash
pixi run python input-data/tensor/chaz/chaz_damage.py check-calibration
pixi run python input-data/tensor/chaz/chaz_damage.py fetch-geometry   # only to bump Natural Earth
pixi run python input-data/tensor/chaz/chaz_damage.py build --scenario ssp245 --period base --variant CRH
pixi run python input-data/tensor/chaz/chaz_damage.py build --all
pixi run python input-data/tensor/chaz/chaz_damage.py verify chaz_damage_fraction_conus_ssp245_base_CRH_points
```

`verify` checks that values stay in [0, 1], that NaN masks agree across
bands, that damage grows with return period, that `ead` sits inside its
envelope, and that the store's manifest record matches the running code's
`ead_convention` and `wind_cap` — a store built before a convention changed
looks self-consistent otherwise. Exits non-zero on any violation.

### Walkthrough

`notebooks/chaz-damage-walkthrough.ipynb` derives all of the above from the
served stores, figure by figure; Part 4 is the validation against the
event-set EAD. It imports `chaz_damage` by walking up from the working
directory, so it runs from anywhere in the repo. Part 4 reads three large
validation files (shared by Simona Meiler, July 2026) that live at
`s3://carbonplan-ocr/ocr-explore/CHAZ/validation/` and are fetched
automatically into a local cache (`~/.cache/ocr/chaz-validation`;
`CHAZ_VALIDATION_DIR` overrides), so the whole notebook runs off S3 access
alone.

### Outputs

37 topozarr pyramid stores (18 per-GCM, 18 median, 1 ERA5), CONUS crop
(−125→−66°E, 24→50.5°N) clipped to NA2, bands `rp_10 … rp_1000` plus
`ead`/`ead_lower`/`ead_upper`:

```
chaz_damage_fraction_conus_{scenario}_{period}_{variant}[_median]_points
chaz_damage_fraction_conus_ERA5_points
```

Each manifest record carries the calibration provenance (`v_half`,
`v_thresh`, `calibration`, `wind_cap`, `ead_convention`, source store).

### Caveats

- **Wind is a proxy for total TC damage.** The calibration target (EM-DAT) is
  total per-event loss, implicitly including surge and rain damage (Eberenz
  Sect. 2.2.3) — expect overlap with any future coastal-flood layer.
- **Per-cell values are expectations.** The function is fit to
  country-aggregated event damages; individual events spread widely around it
  (Eberenz Sect. 3.1.2).
- **NA2 only.** Non-NA2 cells inside the CONUS bbox and open water are
  masked; no US or Canadian cell is dropped. Going beyond NA2 needs a
  per-cell `v_half` keyed to country.
- **The tail is extrapolated, tuned, and capped.** The convention past
  `rp_1000` was chosen to match one ERA5 event-set reference and is untested
  on the future scenarios; it is worth a couple of percent either way. The
  90 m/s cap is a fixed constant, not a physical bound.
- **Vulnerability is held fixed under climate change.** Future-period damage
  reflects hazard change only; Meiler et al. 2025 find exposure growth, not
  hazard change, dominates future TC risk.
- **Ratios are more robust than levels.** `v_half` uncertainty dominates
  absolute values but largely cancels in future-vs-base comparisons made with
  the same function (Meiler et al. 2025).
- **Source usage notes (Meiler et al. 2026).** Multi-model averages are
  recommended (served as the median stores); CRH and SD are bounding cases,
  not to be averaged; over-ocean values are too sparse to use. The source's
  return levels are empirical event rankings with constant extrapolation
  beyond the event set — no distribution fit.

## Accessing the processed stores

The stores are multiscale Zarr v3 pyramids; level `0` is the native grid.

```python
import xarray as xr

base = 's3://carbonplan-ocr/ocr-explore/CHAZ/processed'
wind = xr.open_zarr(f'{base}/chaz_exceedance_intensity_ssp245_fut1_CRH_points/0', consolidated=False)
dmg = xr.open_zarr(f'{base}/chaz_damage_fraction_conus_ssp245_fut1_CRH_points/0', consolidated=False)

# GCM order is fixed; names travel on the coordinate
wind['gcm'].attrs['names']
```

## References

- Eberenz, S., Lüthi, S., Bresch, D. N. (2021). Regional tropical cyclone impact
  functions for globally consistent risk assessments. _NHESS_ 21, 393–415.
  https://doi.org/10.5194/nhess-21-393-2021
- Emanuel, K. (2011). Global warming effects on U.S. hurricane damage. _WCAS_ 3,
  261–268. https://doi.org/10.1175/WCAS-D-11-00007.1
- Meiler, S., Kropf, C. M., McCaughey, J. W., Lee, C.-Y., Camargo, S. J., Sobel,
  A. H., Bloemendaal, N., Emanuel, K., Bresch, D. N. (2025). Navigating and
  attributing uncertainty in future tropical cyclone risk estimates. _Sci. Adv._
  11, eadn4607. https://doi.org/10.1126/sciadv.adn4607
- Meiler, S., Lee, C.-Y., Camargo, S. J., Sobel, A. H. (2026). Global coastal
  wind hazard maps from the CHAZ tropical cyclone model. _Sci. Data_ 13, 136.
  https://doi.org/10.1038/s41597-025-06452-0
- CLIMADA: https://github.com/CLIMADA-project/climada_python
  (`climada.entity.impact_funcs.trop_cyclone`)
