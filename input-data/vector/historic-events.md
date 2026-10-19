# Historic storms, fires, and floods

The "Previous storms", "Previous fires", and "Previous floods" rows in the ocr-web sidebar answer,
for a selected building or point, which recorded events affected it. This
note is the reference for what those artifacts are, how they are built, what
the numbers mean, and what is still open. Build details for each artifact are
in `ibtracs/README.md`, `mtbs/README.md`, and `storm_events/README.md`; the wind-field problem found
along the way is in `ibtracs/climada-translation-artifact.md`.

## Artifacts

All under `s3://carbonplan-ocr/ocr-explore/`, public over HTTPS, with a
provenance record in `manifest.json` written by the script that built them.

| Artifact                            | What it holds                                                                                                                                                                                                                                                                                    | Built by                                         |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| `ibtracs_conus_winds.zarr`          | `wind(storm, lat, lon)`, uint8 m/s, the modeled peak 1-min sustained wind per storm per cell on the CHAZ 1/12 degree grid; 0 below 17.5 m/s. `footprint` holds the same values chunked one storm per chunk for drawing. Storm table in the group attrs and in `ibtracs_conus_winds/storms.json`. | `ibtracs/ibtracs_winds.py`                       |
| `ibtracs_conus_tracks.pmtiles`      | IBTrACS track segments (layer `tracks`, z0 to z9, `promoteId: SID`) for exactly the storms in the wind store, with each segment's recorded wind and category.                                                                                                                                    | `ibtracs/ibtracs_tracks.py build --subset conus` |
| `mtbs_conus_perims.pmtiles`         | MTBS burned-area perimeters clipped to CONUS (layer `mtbs_perims`, z0 to z11, `promoteId: event_id`), verified to hold every perimeter at max zoom.                                                                                                                                              | `mtbs/mtbs_perims.py build --subset conus`       |
| `storm_events_conus_floods.pmtiles` | NOAA Storm Events Flood and Flash Flood reports inside CONUS (layer `floods`, z0 to z10, `promoteId: event_id`), each its reported bounding box or a point, verified to hold every report at max zoom.                                                                                           | `storm_events/storm_events_floods.py build`      |

The wind store is 0.6 MB. Point reads fetch one 16x16-cell chunk holding
every storm; footprint reads fetch one storm.

## Sources

- **IBTrACS v04r01, since-1980 subset** (NOAA NCEI). USA-agency (NHC) fields:
  position, 1-minute maximum sustained wind, central pressure, and from 2001
  the radius of maximum wind and quadrant wind radii. Tracks continue through
  depression and extratropical stages.
- **NOAA Storm Events Database** (NCEI), Flood and Flash Flood events, 1996
  to present, revised monthly. Each report carries two locations that NWS
  Storm Data publishes as the corners bounding the flooded area (NWSI
  10-1605, section 2.4). Locations are referenced to the nearest town's
  center, often only to two decimal places. Before 2005 almost no reports
  have both locations, so the served set effectively starts in 2005 and is
  thin until 2007. Coastal Flood, Lakeshore Flood, and Storm Surge/Tide
  reports are located by forecast zone only and are not included.
- **MTBS burned area boundaries** (USGS/USFS), 1984 to present. Fires above
  1,000 acres in the West and 500 acres in the East; smaller fires are absent.
- **CLIMADA 6.1.0** (Aznar-Siguan and Bresch 2019) for the wind fields, via
  `input-data/tensor/chaz/climada/chaz_events.py`, the same code that builds
  the CHAZ hazard maps in this repo.
- **CHAZ CONUS centroids** (300 arcsec land cells with distance to coast),
  `ocr-explore/CHAZ/hazard/centroids_conus_0300as.hdf5`, shared with the maps.

## Wind footprints: what the number is

The mph shown for a storm at a point is the wind the hazard model attributes
to that storm at that cell, not a measurement. It is built to sit on the same
footing as the Peak winds return-period layer, so a historic storm's wind can
be read against the cell's return-period curve.

Pipeline, per storm:

1. Read the IBTrACS track with CLIMADA (`provider='usa'`, NA and EP basins,
   1980 to present, `estimate_missing=True`), keep storms that carry
   tropical-storm winds within 5 degrees of CONUS.
2. **Normalize inputs to the CHAZ convention.** Discard NHC's central
   pressure, environmental pressure, and radius of maximum wind; estimate
   pressure from wind and position with CLIMADA's fit, fix environmental
   pressure at 1010 hPa, and let CLIMADA estimate the radius from pressure.
   This is exactly what CLIMADA's CHAZ reader hands the model for every
   synthetic track, so the historic storm is what the maps would produce for
   that track and intensity. `--observed-inputs` keeps NHC's values for
   validation runs only; never serve that variant.
3. **Drop nodes below 34 kt** and split the track into contiguous runs of
   strong nodes, modeled separately, with the per-cell maximum taken per
   storm. See "Node rule" below.
4. Crop to the CONUS reach buffer, resample to 30 minutes, and run
   `chaz_events.winds_parallel`: Holland 2008 (`model='H08'`), CLIMADA
   defaults, on the CHAZ centroids. Intensity floor 17.5 m/s, eye radius 300
   km, inland cutoff 1000 km.
5. Round the per-cell peak to whole m/s and scatter onto the served grid.

The maps published by Meiler et al. (2026) were built with the same three
CLIMADA calls (`from_simulations_chaz`, `equal_timestep(0.5)`,
`from_tracks(model='H08')`) on CLIMADA 4.x, which double counted translation
speed; CLIMADA 5.0.0 removed that. The historic footprints are on 6.1.0. Until
the maps are rebuilt on the same version, historic winds for fast-moving
storms sit slightly below the maps' convention.

### What normalization costs

Matching the maps means giving up NHC's observed structure where it exists.
Against the observed-input run, the median wind is unchanged but footprints
move: Charley 2004 at Fort Myers goes from 82 to 134 mph (estimated radius is
too broad for a compact storm), Katrina 2005 at New Orleans from 109 to 88,
Sandy 2012 at Atlantic City from 91 to 65 (estimated radius is too small for
a huge post-tropical storm). Individual values are less faithful to what
happened at an address; the comparison to the hazard layer is the thing that
is consistent.

### Node rule

Best-track nodes with maximum sustained wind below 34 kt are excluded before
the wind step. 34 kt is the NHC and WMO tropical-storm threshold and the wind
model's own intensity floor, and the Holland profile peaks at the node wind,
so such nodes cannot produce tropical-storm-force wind at any cell. Kept in,
they trigger a CLIMADA artifact: for a weak center moving fast, the
translation speed subtracted from a near-zero rotational wind goes negative,
the vortex direction flips, and the motion is added on top, yielding up to
twice the storm's speed. Gustav 2008 gave 62 mph at Chicago from a 20 kt
remnant that way. The rule removes all such inland cases and agrees with a
code-level fix on 96 percent of tropical-storm-force cells; the residual is
storms still at or above 34 kt moving fast near the coast, where a small
overestimate can survive inside the radius of maximum wind.

This is a departure from the maps' inputs, which do carry sub-34 kt nodes
(about a third of CHAZ nodes, tracks ending around 22 kt), confined to nodes
that can only contribute through the artifact. The code-level fix and the
recommended path for the map rebuild are in
`ibtracs/climada-translation-artifact.md`. No clipped output is published.

Interior track stages are still drawn on the map; they are styled faintly
below 34 kt so the modeled and unmodeled parts of a track are visible.

### Sanity checks

At the cell nearest each place, CHAZ-style inputs with the node rule:

| Place            | Top storms (modeled mph)                     | Reported                                            |
| ---------------- | -------------------------------------------- | --------------------------------------------------- |
| Fort Myers FL    | Charley 2004 134, Ian 2022 132, Irma 2017 88 | Ian 150 at Cayo Costa; Charley's core 35 km north   |
| New Orleans LA   | Ida 2021 101, Zeta 2020 90, Katrina 2005 88  | Katrina Cat 1 to 2 in the city                      |
| Rockport TX      | Harvey 2017 108                              | ~110 sustained nearby                               |
| Atlantic City NJ | Bertha 1996 68, Sandy 2012 65                | Sandy ~80 at landfall                               |
| Homestead FL     | Andrew 1992 116                              | 165                                                 |
| Chicago IL       | none                                         | Gustav and Olga remnants at 20 kt                   |
| Niagara Falls NY | Opal 1995 43, Ike 2008 40                    | Both crossed at 35 to 50 kt as extratropical storms |

Expect errors of tens of mph at a point, low for compact intense storms
(Andrew), and no terrain or roughness adjustment: winds are open-terrain
equivalents, as in the maps.

### Comparison to the published CLIMADA dataset

ETH's data API serves `tropical_cyclone_0synth_tracks_150arcsec_historical_USA_1980_2020`,
the same recipe on CLIMADA 3.x with observed inputs. Against our run on shared
cells it has 40 percent more tropical-storm-force cell-events and 2.5 times
the hurricane-force cell-events, with a median of 11 mph higher where both are
nonzero; the double counting explains most of it. It ends in 2020. It is a
useful citation for the method, not a substitute for the artifact.

## Fire perimeters

Relevance is point-in-polygon. The web app fetches the max-zoom tile under
the point from the pmtiles archive directly (pmtiles + vector-tile libraries)
and tests containment with turf, so the answer does not depend on what the
map has rendered. When no perimeter contains the point, the eight neighboring
tiles are read and the nearest perimeter within 15 km is reported. Tiles at
z11 resolve to about 5 m, and `verify --coverage` confirms no perimeter was
dropped for tile size.

## Flood reports

Relevance is distance: a report is near the point when its bounding box (or
point) is within 2 km. Containment alone would almost never match, since
half the boxes are under 1.2 km across, and the location precision of the
reports is about a kilometer anyway. The web app reads the z10 tiles covering
that radius from the pmtiles archive and measures distance to each report's
geometry; boxes crossing tile edges are clipped per tile, so the nearest part
of any box within the radius is always in a tile that is read. Boxes under
100 m on a side are written as points, because tippecanoe otherwise merges
such tiny polygons away even at max zoom.

The flood damage probability layer (Collins et al. 2022) was trained on these
same reports, so the two read together: the reports are where flood damage
was recorded, the layer is where it is likely.

## Web side

- `lib/historic-events/storm-winds.ts`: zarrita point read of one chunk,
  storms with wind > 0 sorted strongest first, with their index into the
  store.
- `lib/historic-events/fire-perimeters.ts`: tile fetch and containment.
- `hooks/useHistoricEvents.ts`: runs the lookup for the selected point under
  the active hazard, regardless of which layer is showing.
- `components/storm-tracks.tsx`, `components/fire-perimeters.tsx`: vector
  overlays, relevant events emphasized, the rest dimmed; two-way hover with the
  list once a point is selected.
- `components/storm-footprint.tsx`: draws one storm's `footprint` slice with
  the zarr layer and the Saffir-Simpson bins of the Peak winds colorbar.
- `lib/historic-events/flood-reports.ts`: tile fetch and distance.
- `components/flood-reports.tsx`: vector overlay, same emphasis and hover.
- `HISTORIC_URLS` in `lib/config.ts` holds the URLs.

## Rebuilding

```
E=~/dev/carbonplan/ocr/.pixi/envs
PYTHONPATH=input-data/tensor/chaz/climada $E/climada/bin/python input-data/vector/ibtracs/ibtracs_winds.py winds
$E/default/bin/python input-data/vector/ibtracs/ibtracs_winds.py store --force
cd input-data/vector/ibtracs && uv run ibtracs_tracks.py build --subset conus --force && uv run ibtracs_tracks.py verify --subset conus
cd ../mtbs && uv run mtbs_perims.py build --subset conus --force && uv run mtbs_perims.py verify --subset conus --coverage
cd ../storm_events && uv run storm_events_floods.py fetch && uv run storm_events_floods.py build --force && uv run storm_events_floods.py verify --coverage
```

The `winds` stage takes about 30 seconds on a laptop. Rebuild the tracks
subset whenever the wind store changes so the two describe the same storms.

## Open items

- **Artifacts are overwritten in place.** A client can hold cached chunks from
  one build and the storm table from another; since storm order changes
  between builds, names would then attach to the wrong winds. Version the
  keys per build and point `HISTORIC_URLS` at a specific version.
- **Map rebuild on CLIMADA 6.1.** Removes the translation double counting
  from the maps and makes the two products the same version. Only the ERA5
  track set is on S3 (40 files, 15.7 GB, a different sample from the
  published maps); the CMIP6 track sets behind the future maps are not, so a
  full rebuild needs them from the CHAZ group.
- **Upstream report** of the translation artifact to CLIMADA, with the fix
  in `climada-translation-artifact.md`. If it lands, the node rule can stay
  and the residual disappears.
- **CHAZ wind averaging.** Confirm `Mwspd` is a 1-minute wind; NHC winds are,
  and CLIMADA's CHAZ reader applies no conversion.
- **Placing storms on the return-period curve** in the panel is now
  legitimate and would make the comparison explicit.
