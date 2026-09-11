# Phantom winds from storm motion in CLIMADA's Holland 2008 wind field

CLIMADA 6.1.0, `climada/hazard/trop_cyclone/trop_cyclone_windfields.py`,
`_compute_windfields`. Found 2026-09-11 while checking the historic storm
footprints (`ibtracs_winds.py`) against NHC best tracks.

## Symptom

Gustav 2008 shows 62 mph at Chicago. NHC's best track has its remnant passing
over Chicago at 20 kt with a central pressure of 1002 hPa, well below
tropical-storm strength. The same value appears with observed inputs and with
CHAZ-style estimated inputs.

## Mechanism

For the H08 and H10 models the per-cell wind is assembled as:

1. `v_ang_norm`: wind magnitude from the Holland profile, driven by the
   pressure deficit. Holland fitted the profile to observed maximum winds, so
   this is a total wind that already includes storm motion.
2. `v_ang_norm -= vtrans_norm * corr`, removing the motion contribution so it
   can be added back as a vector (`corr = min(1, rmax / d)`, the
   Mouton and Nordbeck absorbing factor).
3. The cell's cyclonic tangential unit vector is scaled by `v_ang_norm`.
4. The translation vector times `corr` is added.

Step 2 assumes the profile wind is at least the motion contribution. For a
weak center moving fast it is not: the deficit is clamped to `np.spacing(1)`,
the profile wind is ~0, and `v_ang_norm` becomes about `-vtrans_norm`. The
negative magnitude reverses the tangential vector in step 3, so the cell gets
an anticyclonic wind of `vtrans_norm`, and step 4 adds `vtrans_norm` again.
Where the two align the result is close to `2 * vtrans_norm`; with
translation capped at 30 kt that is about 60 kt, the 62 mph seen at Chicago.

CLIMADA already guards the scalar preparation for the Holland 1980 path with
`np.fmax(0, vmax - vtrans_norm)` (`_vgrad`); the guard was not carried into
the per-cell vector step used by H08 and H10. The 5.0.0 fix for translation
double counting did not address this.

## Evidence

Gustav at a single Chicago centroid, CHAZ-style inputs, `model='H08'`:

| Run                                       | Chicago |
| ----------------------------------------- | ------- |
| As shipped                                | 62 mph  |
| Translation zeroed                        | 0 mph   |
| Rotational wind clipped at 0 after step 2 | 0 mph   |

Full historic set (401 storms, 1980 to 2026), shipped versus clipped:

|                                       | Shipped            | Clipped   |
| ------------------------------------- | ------------------ | --------- |
| Cell-events with tropical-storm winds | 266,480            | 191,828   |
| Cell-events with hurricane winds      | 18,682             | 18,682    |
| Storms with any CONUS winds           | 401                | 391       |
| Chicago                               | Gustav 62, Olga 46 | none      |
| Atlantic City                         | 11 storms          | 7 storms  |
| Fort Myers, New Orleans, Houston      | unchanged          | unchanged |

## Scope

- Historic footprints: about 28 percent of tropical-storm-force cell-events
  and 10 storms are artifacts. They sit along NHC's extratropical remnant
  tracks far inland and where weak, fast storms cross coastal land cells.
  Hurricane-force values are unaffected.
- CHAZ hazard maps: no effect in the interior, because CHAZ tracks end after
  landfall rather than continuing as remnants (one of 87,918 synthetic nodes
  in the ERA5 file 0 subset lies within 1.5 degrees of Chicago, and the served
  return levels there are zero through 1-in-1000). A small effect near the
  coast at the 10-year level. Damage layers are unaffected, since the impact
  function starts at 25.7 m/s.

## Fix

Clip the rotational magnitude at zero after removing the motion contribution:

```python
v_ang_norm[mask_centr_close] = np.fmax(
    0,
    v_ang_norm[mask_centr_close]
    - vtrans_norm_bc[mask_centr_close] * v_trans_corr[mask_centr_close],
)
```

A center whose total wind is entirely its motion then contributes only the
attenuated translation vector, at most 30 kt, which falls under the 34 kt
floor and records as no wind. Storms whose profile wind exceeds their motion
are unchanged.

## Recommendation

Apply the clip in the shared wind step (`chaz_events.winds_parallel`) so the
historic footprints and the rebuilt hazard maps stay identical by
construction, record it in both manifests, and report it upstream to CLIMADA.
Clipping only the historic run would break parity in exactly the regime where
the two differ; leaving both unclipped keeps parity but leaves phantom inland
tropical-storm events in the historic panel.

Clipped intensities for the full historic set are cached at
`~/.cache/ocr/ibtracs/winds/intensity_clipped.npz`; nothing clipped has been
published.
