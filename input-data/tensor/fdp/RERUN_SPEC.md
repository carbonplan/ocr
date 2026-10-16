# FDP validation and updated-damage rerun

Status: proposed, 2026-10-15. Branch reviewed: `origin/Shane98c/fdp` at
`b3a998e`. This is a research and implementation specification, followed by the
validation progress update below.

Update: the initial [local temporal ranking validation](validation-2026-10-15/README.md)
was completed on 2026-10-15 using watershed-matched backgrounds. It does not
complete the land-cover/floodplain-matched reproduction described below.
No retraining or publication has been run. Its inspection of 2024–2025 means
those years are no longer untouched holdouts for decisions informed by this run.

## Decision

An updated-damage rerun is feasible in principle: USGS lists the original
response points, all 11 predictor rasters, and the R model code. The existing
OCR branch only packages the published output raster and samples it onto
buildings. Its `build` command does not retrain anything, and its `verify`
command checks storage integrity rather than predictive performance.

Proceed in stages: validate the frozen published surface on later reports;
reproduce one watershed; evaluate a damage-only refit; then produce a national
refit. Refresh predictor vintages as a separate experiment.

## Evidence and original method

- Collins et al. (2022), methods pp. 2–4: **December 2006–May 2020**, 71,434
  reported flash/riverine flood damage locations. Coastal, storm surge/tide,
  and lakeshore flood event types were excluded. The paper calls this an
  approximately 14-year period; it is not 15 complete calendar years.
- Eighteen separate HUC-2 watershed models, 100 m grid. Pseudo-absences matched
  presence counts by watershed and land-cover class, excluded the combined
  FEMA/EPA 100-year floodplain and cells containing presences (1:1 PA:P).
- Eleven predictors: floodplain, stream distance, elevation, HAND, land cover,
  profile curvature, hydric soils, topographic wetness index, channelization,
  urban road density, and mean annual maximum three-day precipitation.
- Important vintages: 2016 land cover/roads, precipitation averaged over
  2006–2019. Holding these fixed isolates the effect of new response data;
  it does not describe fully updated landscape/climate conditions.
- Released R code uses `ranger` probability forests with 1,000 trees,
  `tuneRanger` (70 iterations, Brier objective), four folds and pairwise-distance
  sampling to address spatial sorting bias. The reported mean AUC is 0.750;
  that is not a guaranteed target for a different temporal evaluation.

Sources: [paper](https://doi.org/10.1088/1748-9326/ac4f0f),
[USGS release](https://doi.org/10.5066/P954TTQN),
[release metadata](https://www.sciencebase.gov/catalog/item/6170694ed34ea36449a67ef7?format=json),
[released R code](https://www.sciencebase.gov/catalog/file/get/6170694ed34ea36449a67ef7?f=__disk__f2%2Ffa%2F52%2Ff2fa5277d1374766fce283b8bb47d2ab73b767b6).

## Data readiness

The live [NOAA bulk index](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/)
lists 2025 and partial 2026 data. A local inventory of
`StormEvents_details-ftp_v1.0_d2025_c20260819.csv.gz` found 7,654 `Flood` or
`Flash Flood` rows spanning all twelve months, all with nonempty begin-coordinate
strings. These are raw national event rows, not vetted CONUS damage locations;
this count is not a validation sample size or evidence of complete reporting.

Use **2025 as the proposed last full year**, subject to coverage QA. Keep 2026
out of headline comparisons. NOAA revises older years as well as recent years;
record the filename creation date, retrieval time and SHA-256 of every input.
See the [NOAA README](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/README).

USGS metadata lists about 18.0 GB of compressed predictor archives. The small
README and R script were retrieved successfully. Response archive links returned
HTML through the listed manager URLs; a name-based presence download returned 404. Actual retrieval of response archives and predictors remains a readiness
gate. Do not equate catalog availability with verified downloadable inputs.
The code also expects `HUC2_CONUS_proj`, absent from the top-level file list;
obtain and version the matching watershed boundaries.

Access update (2026-10-16): the browser download flow requires a human CAPTCHA.
The floodplain predictor was successfully downloaded and inspected locally
(100 m, EPSG:5070); the remaining predictors have not been acquired. The latest
2026 NOAA details file, `StormEvents_details-ftp_v1.0_d2026_c20260918.csv.gz`,
contains January–June only, including 2,473 raw Flood/Flash Flood event rows.
A refit using all currently available reports would therefore end in June 2026,
not December. Keep the completed 2020–2025 validation snapshot unchanged.

## Phase 1: frozen-map temporal validation

Question: does the original surface rank subsequently reported damage locations
above comparable background locations?

1. Freeze the published native-resolution raster and its checksum. Sample
   level 0 directly at report locations, not the building table or overviews.
   First verify that a sample of served level-0 cells matches the source TIFF.
2. Ingest NOAA details and locations for June 2020–December 2025. Join on
   `EVENT_ID`; retain `EPISODE_ID`, location index, event dates/types and damage,
   injury/fatality and narrative fields. Preserve raw records and provenance.
3. Recover the original point-construction and inclusion rules from the
   supplementary methods and released response data before declaring parity.
   Resolve begin versus end versus multiple reported locations, monetary-damage
   filtering, and repeated locations. Do not assume `DAMAGE_PROPERTY > 0`:
   the paper's damage definition also includes injury, death and disruption.
   Do not multiply event-level dollar amounts across location rows.
4. Filter to CONUS and the original two event types. Audit invalid coordinates,
   missing values, duplicate event/location records, HUC assignment and raster
   coverage. Report each exclusion, including nodata, by year and HUC.
5. Primary positive summary: unique 100 m damaged cells within the evaluation
   period. Also report event/location-weighted results, and separate cells seen
   in original training from newly observed cells. Freeze this choice before
   comparing scores. A future report at an old site is valid temporal evidence
   but does not establish spatial generalization.
6. Sample reproducible evaluation backgrounds using the paper's HUC/land-cover
   matching and floodplain exclusions. Exclude known presence cells through the
   evaluation cutoff. Repeat across fixed seeds to measure sampling sensitivity.
   As a separate robustness analysis, use land-cover/HUC-matched backgrounds
   without the floodplain exclusion: the original exclusion makes floodplain
   membership an especially favorable predictor.
7. Report per-year and per-HUC ROC AUC, PR AUC with explicit sampled prevalence,
   score distributions, and recall/lift in the highest-scoring 10% and 20% of a
   defined eligible land area. Report pooled and equal-HUC summaries separately.
   Stratify by event type, land cover, floodplain class and old/new locations.
   Bootstrap by episode/spatial block rather than treating adjacent points as
   independent. Include stream-distance and floodplain-only baselines where
   inputs are available.
8. Do not tune a threshold on this test. For binary recall, sample the published
   binary map if obtainable; do not substitute a global threshold of 0.5.
   Any refit thresholds must be learned within development folds.

Outputs: input manifest; exclusions/counts table; scored presence/background
Parquet; metrics with uncertainty; compact evaluation report and maps of misses.
This tests later _reported damage_, not all actual flooding or true absences.
The pseudo-absence design does not establish population calibration; Brier and
reliability diagrams, if included, describe the sampled classification task.

## Phase 2: reproducible pilot and damage-only comparison

Start with one HUC-2 (proposed HUC 03, South Atlantic-Gulf), then add a
contrasting watershed before scaling. Benchmark actual resource usage.

### Reproduction gate

- Retrieve the original points, rasters, boundaries and supplementary methods.
  Verify CRS, transform, shape, mask and categorical encodings; extract a
  versioned training table. Reconcile counts against 71,434 before claiming
  that a reconstructed NOAA extraction matches the study.
- Preserve original code as a reference, parameterize paths and HUC selection,
  and pin a runnable R environment. Audit legacy geospatial dependencies.
  Prefer retaining `ranger` for the initial reproduction; a Python forest is
  not automatically an equivalent implementation.
- Document code issues: the preliminary `testbackg` filter uses
  `complete.cases(testpres)`; final background selection uses `group` while CV
  uses `group_backg`; folds/tuning are not comprehensively seeded. The script
  chooses the highest-AUC fold and trains the final map on a subset rather than
  an explicit all-data refit. Separate faithful reproduction from corrections.
- Save all folds, tuning results, thresholds and model objects. Assess raster
  rank agreement and absolute differences with the published map. Exact
  bitwise reproduction is not promised given missing seeds/environment details.

### Comparative experiment

Keep the original predictor rasters fixed. Use a common locked evaluation set:

| Model                     | Training responses          | Test      |
| ------------------------- | --------------------------- | --------- |
| Published baseline        | Dec 2006–May 2020           | 2024–2025 |
| Reproduced baseline       | Original released responses | 2024–2025 |
| Updated development model | Dec 2006–Dec 2023           | 2024–2025 |

Use June 2020–2023 for development diagnostics. Treat the broad Phase 1 report
as descriptive: if its 2024–2025 findings influence model choices, that period
is no longer a pristine confirmatory holdout. For a confirmatory comparison,
reserve those years before inspecting their outcomes, or later collect an
untouched future period.

Regenerate training pseudo-absences for each training window with the original
stratification; new damage can invalidate old pseudo-absence cells. Training
construction must use training-window information only. Evaluation background
construction may use test labels to exclude test-positive cells. Use independent
seeds for training and evaluation samples and measure sensitivity to resampling.
Group repeated locations and related events in development folds; include
spatial blocking. Tune only within development data, choose thresholds from
out-of-fold predictions, then refit on all permitted training data.

Compare both refits with the same corrected modeling procedure so algorithm
corrections are not attributed to added damage data. Use a fixed old-data
snapshot where possible; quantify revisions to historical NOAA records as a
separate change from newly appended years.

Proceed to national production if input parity and reproducibility are
documented, the pilot resource profile is workable, and evaluation shows useful
ranking with understood regional failures. A refit need not outperform the old
map; report that outcome rather than automatically replacing it.

## Phase 3: full national rerun

After the evaluation protocol is complete, fit 18 HUC-2 models on December
2006–December 2025 and regenerate pseudo-absences for that full period. This
production model consumes the earlier test period; retain the frozen development
models and their test results, and do not call a 2025 score out-of-sample for
the production fit.

The expanding window is about 19 years rather than the paper's approximately
14 years. Label it explicitly. If equal-duration comparisons matter, add a
2012–2025 sensitivity fit; raw score changes otherwise mix added observations,
window length, reporting changes and model changes. Do not annualize scores
using an independent-year formula. Balanced, constrained pseudo-absence
sampling does not yield a calibrated annual probability of building loss.

Predict in bounded-memory raster windows, mosaic on the original EPSG:5070
100 m grid, and verify seams, masks, range and factor levels. Eleven full-grid
float32 arrays alone are roughly 67 GB before model and processing overhead;
do not load the whole stack eagerly. Stage predictors once near compute, run
HUC jobs with explicit thread limits and resumable outputs, and measure pilot
runtime/RAM/disk before giving a national cost estimate. The existing README's
12 GB packaging estimate is not a training-resource estimate.

Publish only to a versioned experimental prefix such as
`ocr-explore/FDP/runs/<run-id>/`. Extend `fdp.py` to accept an explicit source
raster and destination prefix; the current source and pruning destination are
hard-coded. Reuse pyramid/building sampling after raster QA, but keep the
published original intact. Record response dates, predictor vintages, code and
environment versions, model parameters, sample seeds, thresholds and source
hashes in the run manifest. Report dropped building counts and nodata coverage.

## Proposed implementation units

These are new components to implement, not existing executable commands:

1. `noaa_reports.py`: immutable downloads, normalization, location policy and QA.
2. `validate_fdp.py`: raster sampling, reproducible backgrounds and evaluation.
3. `prepare_training.py`: grid-aligned feature extraction, HUC assignments,
   pseudo-absence generation and frozen splits.
4. `train_fdp.R`: parameterized HUC-specific fitting, tuning and saved models.
5. `predict_fdp.R`: windowed prediction and validated national mosaic.
6. Existing `fdp.py` / `fdp_buildings.py`: versioned ingestion of selected outputs.

Targeted checks should cover NOAA event/location joins, duplicate handling,
CRS/cell indexing, forbidden pseudo-absence cells, split leakage, categorical
levels, and equivalence between model predictions and raster output. A small
end-to-end HUC subset must pass before a cloud-scale run.

## Later predictor-refresh experiment

Update land cover, road density and precipitation summaries first as a distinct
version; assess other layers for meaningful changes. Preserve variable
definitions, categorical mappings and grid alignment. Establish an explicit
prediction date and prevent future predictor data from leaking into temporal
tests. Compare damage-only and damage-plus-predictor updates on identical
held-out data. Current-condition and climate-scenario products would require
additional design beyond this historical rerun.
