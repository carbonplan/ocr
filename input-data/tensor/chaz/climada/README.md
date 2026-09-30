# CHAZ event set: testing the EAD tail against the full ERA5 storm catalogue

The served CHAZ stores are six return levels per cell, and `chaz_damage.py`
integrates them into an expected annual damage that extrapolates past the
1000-year band (walkthrough §5). The scripts here rebuild the ERA5 event set
for CONUS in CLIMADA so the damage–frequency curve is exact per cell, and
the extrapolation convention can be scored where it is applied, re-scored at
the scenario stores' sample size, and stressed under warming-like shifts.
Part 5 of the walkthrough reads the results below.

The hazard is built the way the published maps were (Meiler et al. 2026,
[CHAZ-hazard-maps](https://github.com/simonameiler/CHAZ-hazard-maps)):
CLIMADA's CHAZ reader, half-hour resampling, Holland (2008) winds with
CLIMADA's defaults, the published 300 arcsec land cells, one frequency per
basin (IBTrACS annual count over simulated storms with tropical-storm winds
in the basin), and empirical exceedance ranking with constant extrapolation.
It does not reproduce her numbers, and is not meant to: the published maps
carry winds from a CLIMADA version that double-counted translational
velocity (see `check` below), so our levels sit 10–15% under the Dryad
points by construction. A small residual sample difference is discussed
there too.

## Environment

CLIMADA pins Python below 3.13, so it lives in its own pixi environment:

```bash
pixi install -e climada
pixi run -e climada python input-data/tensor/chaz/climada/chaz_events.py --help
```

The Coiled software environment `ocr-climada` is that environment exported
for `linux-aarch64`:

```bash
pixi workspace export conda-environment -e climada -p linux-aarch64 \
  | pixi run -e climada python .github/scripts/export_pixi_to_conda.py \
      --stdin --output environment-climada.yaml --coiled-name ocr-climada --no-include-local-code
```

`chaz_tail_test.py` needs only numpy and `chaz_damage.py`, and runs in the
default environment.

Simona Meiler's own files behind the published ERA5 maps sit under
`s3://carbonplan-ocr/ocr-explore/CHAZ/published/`: the frequency-corrected
global hazard (`TC_global_0300as_CHAZ_ERA5_freq-corr.hdf5`, 33 GiB, the
event set the maps were ranked from), the exceedance impact output
(`TC_global_0300as_CHAZ_ERA5_corrected_exceedance_impact.nc`), and the
LitPop 2020 exposure (`litpop_0300as_2020_global.hdf5`). Section 4 covers
what is in the hazard file and how it is cropped to CONUS.

## 1. Hazard (`chaz_events.py`)

Everything lands under `s3://carbonplan-ocr/ocr-explore/CHAZ/hazard/ERA5/`
and is cached locally in `~/.cache/ocr/chaz-events` (`CHAZ_EVENTS_DIR`).

```bash
E="pixi run -e climada python input-data/tensor/chaz/climada/chaz_events.py"
$E centroids                       # the Dryad ERA5 points in the CONUS box, 127,602 cells
$E events --files 0-9              # per-file event tables and CONUS track subsets
$E hazard --file 0 --members 0-7   # one chunk locally (about 8,000 tracks, ~3 min on 8 cores)
$E batch --files 0-9               # the remaining chunks on Coiled, 4 members per task
$E levels --files 0-9 --years 1981-2019 --tag era5_10f
$E check --tag era5_10f            # ratio table and figure against the Dryad points
```

- `events` scans each track file once and records, per (member, storm):
  year, peak wind, and whether it carries tropical-storm winds within 5° of
  the CONUS box and in which basin. Only reaching storms go into the subset
  file CLIMADA reads. The writer blanks winds past a storm's end, since the
  raw fill dates otherwise survive as a spurious node and break resampling.
- `hazard` is one Coiled task: read a file's subset for a run of members,
  crop each track to the 5° buffer plus two nodes, resample, compute Holland
  winds on the CONUS cells across our own process pool, and write a CLIMADA
  HDF5. Tracks go through `TropCyclone.from_tracks` fifty at a time to keep
  memory near 3 GB. About 0.19 core-seconds per track; the ten-file set is
  roughly 370,000 tracks, or 20 CPU-hours. Distance to coast comes from the
  data API's 150 arcsec centroids, since the NASA raster CLIMADA expects is
  no longer hosted.
- `batch` submits chunks through `coiled.batch.run` on ARM spot VMs
  (m8g.xlarge, us-west-2). Each task pulls the script from our bucket,
  uploads its log to `logs/` every minute, and skips chunks already on S3.
  The 96 tasks for ten files ran in 29 minutes on 25 VMs.
- `levels` joins the chunks (deduplicating overlapping events), applies the
  year window, sets frequencies from the basin counts over the members held,
  and writes `sets/<tag>.hdf5` plus the six return levels and the 33 and 50
  m/s return periods to `levels/<tag>.nc`. Because the weight is an observed
  count over a simulated count, only the storm sample matters, which lets
  the tail test subsample freely.
- `check` compares our levels to the published ERA5 points as a median
  ratio per band, binned by the published 100-year wind. On the ten-file
  1981–2019 set (206,770 events) our winds sit a uniform 10–15% below the
  published maps in every band, and the 33 and 50 m/s return periods are
  2–4 times longer.

That gap is CLIMADA, not the tracks. Before CLIMADA 4.1.0 the Holland wind
models added the storm's translational velocity twice (fixed in PR #833,
January 2024); on 150 of our tracks, 3.3.2 and 4.0.1 give winds 12–16%
higher than 4.1.0 through 6.1.0. Simona Meiler confirmed (September 2026)
that the ERA5 fields were built on ~3.3.0-dev and the CMIP6 fields on
~4.0.0-dev, and the 2025 frequency correction did not touch winds. Every
published map therefore carries the double count. The tail test is internal
to our set and unaffected. A smaller residual remains: the published weights
imply ~18% more storms per basin than the shared files hold, worth a
question to the authors but worth only a percent or two in return levels.

## 2. Tail test (`chaz_tail_test.py`)

```bash
T="pixi run python input-data/tensor/chaz/climada/chaz_tail_test.py"
$T --tag era5_10f check                 # ranking reimplementation vs the CLIMADA levels
$T --tag era5_10f decompose             # truth vs reconstruction by T range, per cell
$T --tag era5_10f subsample --draws 50  # at the CMIP6 stores' design, 10 files x 8 members x 20 yr
$T --tag era5_10f perturb               # frequency x0.7/x1.3, wind x1.05/x1.10
```

Truth is the frequency-weighted sum of the impact function over events at
unit exposure, with the NA2 `v_half` of the chosen calibration (default
`TDR1.0`, the product's). Reconstruction is `chaz_damage.ead_from_levels`
on the six ranked levels in three tail variants: the shipped slope
extension, a flat tail, and truncation at rp_1000. Both sides are split into
T < 10, 10–1000 and > 1000 years.

`subsample` re-levels random subsets of (file, member) pairs with the same
ranking and weight rules, so each draw is a self-contained event set like
the ones behind the scenario stores. `perturb` rescales weights and winds in
place; scaling wind is a first-order stand-in for a hotter track set.

Outputs go to `hazard/ERA5/tail/`, named `<tag>_[<calibration>_]<experiment>`
with no calibration tag for TDR1.0 (the `CAL_TAG` convention of
`chaz_damage.py`); the parquets also carry a `calibration` column.

## 3. Results on the ten-file ERA5 set (`era5_10f`)

206,770 events reaching CONUS, 1981–2019, NA2 `v_half` 89.2 m/s (TDR1.0),
rarest resolved return period 18,344 years. Ratios are reconstruction over
truth; "weighted" is LitPop 2020 value over US cells.

|                          | truth tail share (T > 1000) | slope (shipped) | flat  | truncate |
| ------------------------ | --------------------------- | --------------- | ----- | -------- |
| unit exposure, all cells | 22%                         | 1.088           | 0.907 | 0.780    |
| LitPop-weighted, US      | 24%                         | 1.058           | 0.907 | 0.767    |
| rp_100 wind 26–30 m/s    | 57%                         | 1.126           | 0.690 | 0.441    |
| rp_100 wind 40–45 m/s    | 15%                         | 1.013           | 0.956 | 0.850    |
| rp_100 wind 50–55 m/s    | 7%                          | 1.012           | 0.994 | 0.940    |

Truth sits between the two conventions everywhere, closer to the slope
extension where damage lives. Within the bands (10–1000 yr) the
reconstruction is 1% low everywhere; the whole disagreement is the tail.
Warming-like rescaling (frequency ×0.7 and ×1.3, wind ×1.05 and ×1.10)
keeps the weighted slope ratio between 1.03 and 1.07 and the flat ratio
between 0.87 and 0.95, so the convention transfers to shifted distributions
at this sample size.

**At the scenario stores' sample size it does not.** Fifty draws at the
CMIP6 design (10 files × 8 members × 20 years, 1,600 simulated years, rarest
resolved return period ~2,000 years), each re-weighted and re-ranked:

| ratio to the draw's own truth  | median | IQR         |
| ------------------------------ | ------ | ----------- |
| slope (shipped), unit exposure | 1.248  | 1.227–1.260 |
| slope, LitPop-weighted         | 1.184  | 1.165–1.200 |
| flat, unit exposure            | 1.000  | 0.995–1.005 |
| flat, LitPop-weighted          | 0.996  | 0.989–1.001 |
| truncate, unit exposure        | 0.847  | 0.844–0.850 |

With 1,600 simulated years the 1000-year band rests on the top one or two
events and CLIMADA's constant extrapolation holds the curve flat beyond
them, so the true tail share is small (12%) while the slope extension keeps
climbing from a segment set by the rarest events.

**Verdict.** The slope extension is 6% high on ERA5 (18,000 simulated
years) and 18–25% high on the CMIP6 stores the product ships, where a flat
tail is unbiased. A flat tail past rp_1000 is the better single convention:
worst case 9% low on the ERA5 reference against 22% high on every scenario
store. Changing `ead_convention` in `chaz_damage.py` means rebuilding the
damage, v2 and buildings products, and `verify` will flag stores built under
the old convention.

Caveats: truth is our own event set, the draws use ERA5 storms at a GCM
sample size rather than GCM storms, and the tail contribution scales with
the impact function's curvature, so the ratios hold for NA2 at this `v_half`
and would tighten with a flatter function.

## 4. The published event set (`chaz_published.py`)

```bash
P="pixi run -e climada python input-data/tensor/chaz/climada/chaz_published.py"
$P submit                 # crop + match on Coiled, one VM each (~7 min crop, ~3 h Dryad check)
$P tailset                # set, levels and events files for the tail test, tag `published`
T="pixi run python input-data/tensor/chaz/climada/chaz_tail_test.py"
$T --tag published check  # ranking reimplementation vs the Dryad points
$T --tag published decompose
$T --tag published subsample --draws 50
```

The hazard file is one catalogue of 1,395,323 storms (10 track files × 40
members, 1981–2019, every track with wind) stored six times, once per
basin. Each copy carries that basin's constant frequency, one number per
basin, and winds only on that basin's cells; for CONUS the split is at
100°W, East Pacific storms west of it and North Atlantic storms east of
it. Dividing the IBTrACS annual counts by her frequencies gives the storm
counts she corrected to: 207,708 in the North Atlantic (4.8% above our
count over ten files) and 319,096 in the East Pacific (20% above). The
sample size matches ours; the remainder is how "in the basin" is decided.
The file is 35 GB of unchunked, uncompressed HDF5 (two CSR matrices, the
`fraction` one all ones), so `crop` runs on a VM in-region: it keeps the
128,242 columns that are CONUS cells and the 178,255 events with wind
there, and checks CLIMADA's ranking on the crop against the Dryad points.
The published values are reproduced to 4e-6 m/s in every band, so the
Dryad points are the crop's own six levels and serve as the levels file
for the tail test. Ranking 178,255 events on 128,242 cells took CLIMADA
2.8 hours; the tail test's numpy ranking does the same in minutes.

The tracks are not the ones she shared. `match` splits her catalogue into
its 400 member runs (the year sequence restarts at each member, and the
storm-date overlap between consecutive runs drops at every 40th, so the
order is file, member, storm) and tests each against all 40 shared track
files two ways: by the event dates CLIMADA's reader would assign (the first
node with a valid wind in that member, storms selected by the year of node
0), and, independently of dates, by the multiset of the 40 per-member storm
counts of each file against each of her ten file groups. No run matches
any member of any file, no file's member counts match any of her groups,
and the best date overlap is 36%, the chance level. Her February 2023
catalogue is a different sample from the June 2024 files, so a per-storm
comparison of her pre-4.1 winds with current CLIMADA's needs her original
tracks.

### Results on the published set

The same tests as section 3, with her catalogue as truth and the Dryad
points as the six levels, so the ratios score the ERA5 store exactly as
served. 178,254 events with CONUS wind, rarest resolved return period
19,232 years (median over cells).

|                          | truth tail share (T > 1000) | slope (shipped) | flat  | truncate |
| ------------------------ | --------------------------- | --------------- | ----- | -------- |
| unit exposure, all cells | 18%                         | 1.049           | 0.930 | 0.823    |
| LitPop-weighted, US      | 17%                         | 1.024           | 0.942 | 0.837    |
| rp_100 wind 26–30 m/s    | 58%                         | 1.128           | 0.682 | 0.426    |
| rp_100 wind 40–45 m/s    | 16%                         | 1.012           | 0.949 | 0.838    |
| rp_100 wind 50–55 m/s    | 8%                          | 1.010           | 0.990 | 0.931    |

Within the bands the reconstruction is under 1% low everywhere. The
bracket is tighter than on our rebuild: where damage lives the slope
extension is 2% high and a flat tail 6% low. Rescaling frequency ×0.7/×1.3
and wind ×1.05/×1.10 keeps the weighted slope ratio at 1.02–1.03 and the
flat ratio at 0.92–0.98.

Fifty draws at the CMIP6 design (10 files × 8 members × 20 years), each
re-weighted by the fraction of simulated years drawn and re-ranked:

| ratio to the draw's own truth  | median | IQR         |
| ------------------------------ | ------ | ----------- |
| slope (shipped), unit exposure | 1.137  | 1.130–1.142 |
| slope, LitPop-weighted         | 1.088  | 1.084–1.094 |
| flat, unit exposure            | 0.988  | 0.983–0.993 |
| flat, LitPop-weighted          | 0.996  | 0.992–0.999 |
| truncate, unit exposure        | 0.871  | 0.863–0.876 |

The true tail share drops to 11% at that sample size and the slope
extension overshoots it, by less than on our rebuild (9% weighted against
18%) since her winds sit higher on the impact function, but in the same
direction. The verdict of section 3 stands on the served data: a flat tail
past rp_1000 is 2–6% low on the ERA5 store and unbiased at the scenario
stores' sample size, while the slope extension is 2% high on ERA5 and
9–14% high at the scenario sample size.
