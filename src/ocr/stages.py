"""Per-risk output settings and the shared stages: buildings, then regions."""

import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import xarray as xr

from ocr.buildings import PointSamplingStrategy, sample_buildings
from ocr.config import OutputPaths, Risk, StorageConfig
from ocr.lod import write_buildings_lod
from ocr.regions import RegionLevel
from ocr.regions.layout import write_level
from ocr.regions.stats import Hist, write_region_stats

# fire_wind_risk_regional_aggregator.py (fire's rps hist; it computed no bp hists).
FIRE_RPS_BINS = (0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.5, 1, 3, 100)
# fdp is a 0-1 probability (sampled 2026-10-30: continuous, no zeros, ~1/3 of cells at 1).
FLOOD_FDP_BINS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def positive(*cols: str) -> str:
    """SQL predicate for `lod_shown`: buildings whose every col is above 0.

    Buildings on raster nodata keep NaN (`sample_buildings`), and DuckDB treats NaN as larger than
    any number, so a bare `col > 0` would let nodata buildings into the coarse-zoom map.
    """
    return ' AND '.join(f'({c} > 0 AND NOT isnan({c}))' for c in cols)


@dataclass(frozen=True)
class RiskOutputs:
    hists: Sequence[Hist]  # region histograms, `write_regions`
    lod_shown: str  # SQL predicate on building columns: what the coarsest zooms sample, `write_buildings_lod`
    lod_carry: Sequence[
        str | tuple[str, str]
    ]  # columns map tiles carry for styling, `write_buildings_lod`


RISKS = {
    Risk.FIRE: RiskOutputs(
        hists=[Hist(['rps_2011', 'rps_2047'], FIRE_RPS_BINS)],
        lod_shown=positive('rps_2011', 'rps_2047'),
        # half precision: the map colours by bins; selecting a building reads full precision by row
        lod_carry=[('rps_2011', 'halffloat'), ('rps_2047', 'halffloat')],
    ),
    # ponytail: even 0.1 bins from six city windows; revisit with a CONUS quantile pass (in-region Coiled).
    Risk.FLOOD: RiskOutputs(
        hists=[Hist(['fdp'], FLOOD_FDP_BINS)],
        lod_shown=positive('fdp'),
        lod_carry=[('fdp', 'halffloat')],
    ),
    # TODO: hurricane bins, shown predicate and carried columns; empty hists write building_count only.
    Risk.HURRICANE: RiskOutputs(hists=[], lod_shown='true', lod_carry=[]),
}


def write_buildings(
    ds: xr.Dataset,
    out: OutputPaths,
    *,
    blocks: str | Path | None = None,
    source: str | Path | None = None,
    point_strategy: PointSamplingStrategy = PointSamplingStrategy.CENTROID,
    temp_dir: str | Path | None = None,
    **lod_kwargs,
) -> None:
    """Sample `ds` at every CONUS building, tag its block and write `out.buildings`
    with LOD overviews (`out.risk`'s shown/carry) and `version` metadata. `blocks` and `source` go to `sample_buildings`;
    `temp_dir` holds the staged file and LOD spill (on Coiled, a large volume); `lod_kwargs` go
    to `write_buildings_lod`. Region stats then read that file (`write_regions`).
    """
    h = RISKS[out.risk]
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp:
        # geoparquet-lod reads its source several times, so sampling stages a local file.
        src = Path(tmp) / 'buildings.parquet'
        sample_buildings(
            ds, src, blocks=blocks, source=source, point_strategy=point_strategy, temp_dir=tmp
        )
        write_buildings_lod(
            src,
            out.buildings,
            shown=h.lod_shown,
            carry=h.lod_carry,
            metadata={'version': f'v{out.version}'},
            temp_dir=tmp,
            **lod_kwargs,
        )


def write_regions(
    out: OutputPaths,
    levels: Iterable[RegionLevel] = tuple(RegionLevel),
    *,
    temp_dir: str | Path | None = None,
) -> None:
    """Write region stats of `out.buildings` (`write_buildings`) per TIGER level in the
    web layout (`ocr.regions.layout`) under `out.regions`, with `out.risk`'s hists.
    `temp_dir` holds the staged stats and layout files (on Coiled, a large volume).
    """
    hists = RISKS[out.risk].hists
    buildings = StorageConfig().uri(out.buildings)
    shown = [f'{v}_{s}' for h in hists for v in h.variables for s in ('median', 'hist')]
    # ponytail: one buildings scan per level, needed for exact medians; approx_quantile on one block pass rolled up if too slow.
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp:
        for level in levels:
            stats = Path(tmp) / f'{level}.parquet'
            write_region_stats(buildings, level, hists, stats)
            write_level(stats, level, out.regions, shown, temp_dir=tmp)
            stats.unlink()
