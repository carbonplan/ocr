"""Per-risk output settings and the shared stages: etl, score, pyramid, buildings, regions."""

import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any, get_type_hints

import icechunk
import obstore
import xarray as xr
from pydantic import BaseModel, ConfigDict, create_model
from topozarr import create_pyramid

from ocr.buildings import CONUS, PointSamplingStrategy, sample_buildings
from ocr.config import OutputPaths, Risk, StorageConfig
from ocr.lod import write_buildings_lod
from ocr.multiscales import write_multiscales
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
class RiskSpec:
    # Region histograms, `write_regions`.
    hists: Sequence[Hist]
    # SQL predicate on building columns: what the coarsest zooms sample, `write_buildings_lod`.
    lod_shown: str
    # Columns map tiles carry for styling, `write_buildings_lod`.
    lod_carry: Sequence[str | tuple[str, str]]
    # Store CRS, `write_multiscales`. None -> no pyramid.
    crs: str | None = None
    # None -> no such stage for this risk (`unimplemented`).
    # (out, overwrite); writes store.icechunk itself when the risk has no score (flood).
    etl: Callable[[OutputPaths, bool], None] | None = None
    # Writes store.icechunk group "0".
    score: Callable[[OutputPaths], None] | None = None


def _flood_etl(out: OutputPaths, overwrite: bool) -> None:
    # Lazy: flood ETL deps (rioxarray) live in the flood extra.
    from ocr.risks.flood.etl import flood

    flood.run(out, overwrite)


RISKS = {
    Risk.FIRE: RiskSpec(
        hists=[Hist(['rps_2011', 'rps_2047'], FIRE_RPS_BINS)],
        lod_shown=positive('rps_2011', 'rps_2047'),
        # half precision: the map colours by bins; selecting a building reads full precision by row
        lod_carry=[('rps_2011', 'halffloat'), ('rps_2047', 'halffloat')],
    ),
    # ponytail: even 0.1 bins from six city windows; revisit with a CONUS quantile pass (in-region Coiled).
    Risk.FLOOD: RiskSpec(
        hists=[Hist(['fdp'], FLOOD_FDP_BINS)],
        lod_shown=positive('fdp'),
        lod_carry=[('fdp', 'halffloat')],
        crs='EPSG:5070',
        etl=_flood_etl,
    ),
    # TODO: hurricane bins, shown predicate and carried columns; empty hists write building_count only.
    Risk.HURRICANE: RiskSpec(hists=[], lod_shown='true', lod_carry=[]),
}


def write_buildings(
    ds: xr.Dataset,
    out: OutputPaths,
    *,
    blocks: str | Path | None = None,
    source: str | Path | None = None,
    point_strategy: PointSamplingStrategy = PointSamplingStrategy.CENTROID,
    bbox: tuple[float, float, float, float] = CONUS.bbox,
    temp_dir: str | Path | None = None,
    **lod_kwargs,
) -> None:
    """Sample `ds` at every CONUS building, tag its block and write `out.buildings`
    with LOD overviews (`out.risk`'s shown/carry) and `version` metadata. `blocks`, `source` and `bbox`
    (a dev subset; regions then cover only it) go to `sample_buildings`;
    `temp_dir` holds the staged file and LOD spill (on Coiled, a large volume); `lod_kwargs` go
    to `write_buildings_lod`. Region stats then read that file (`write_regions`).
    """
    h = RISKS[out.risk]
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp:
        # geoparquet-lod reads its source several times, so sampling stages a local file.
        src = Path(tmp) / 'buildings.parquet'
        sample_buildings(
            ds,
            src,
            blocks=blocks,
            source=source,
            point_strategy=point_strategy,
            bbox=bbox,
            temp_dir=tmp,
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
    levels: Sequence[RegionLevel] = tuple(RegionLevel),
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


def _delete_prefix(prefix: str) -> None:
    store = StorageConfig().object_store()
    for batch in obstore.list(store, prefix=prefix):
        obstore.delete(store, [m['path'] for m in batch])


# Stages take (spec, out, overwrite, **util kwargs). Storage is the hand-off, so each reads what
# the previous one wrote; run in this order.
def etl(spec: RiskSpec, out: OutputPaths, overwrite: bool) -> None:
    assert spec.etl is not None  # `unimplemented` preflight
    spec.etl(out, overwrite)


def score(spec: RiskSpec, out: OutputPaths, overwrite: bool) -> None:
    assert spec.score is not None
    if overwrite:
        _delete_prefix(out.store)
    spec.score(out)


def _store(out: OutputPaths) -> icechunk.Repository:
    return icechunk.Repository.open(StorageConfig().icechunk_storage(out.store))


def pyramid(spec: RiskSpec, out: OutputPaths, overwrite: bool, **kw: Any) -> None:
    # A rerun with the same settings replaces the levels; overwrite drops them first, base kept.
    assert spec.crs is not None
    write_multiscales(_store(out), spec.crs, overwrite=overwrite, **kw)


def buildings(spec: RiskSpec, out: OutputPaths, overwrite: bool, **kw: Any) -> None:
    # One file, replaced by obstore.put; overwrite unused.
    session = _store(out).readonly_session('main')
    ds = xr.open_zarr(session.store, group='0', consolidated=False, chunks=None)
    write_buildings(ds, out, **kw)


def regions(spec: RiskSpec, out: OutputPaths, overwrite: bool, **kw: Any) -> None:
    # Files replaced by obstore.put. overwrite also drops old per-state/county files of the levels run
    # (a bbox run writes fewer); levels not run are kept. nation/states are single files, so no-ops.
    if overwrite:
        for level in kw.get('levels', tuple(RegionLevel)):
            _delete_prefix(f'{out.regions}/{level}')
    # ponytail: all levels in one job; fan out per TIGER level if the CONUS timing runs need it.
    write_regions(out, **kw)


# Run order. Data flow under output/<risk>/v<version>/:
#   etl -> [score, fire only] -> store.icechunk -> pyramid
#                                               -> buildings -> vector/buildings.geoparquet -> regions
# pyramid and buildings are independent: both read only store.icechunk.
# Adding a stage, dependency or risk: checklist in docs/how-to/data-pipeline.md.
STAGES: dict[str, Callable[..., None]] = {
    'etl': etl,
    'score': score,
    'pyramid': pyramid,
    'buildings': buildings,
    'regions': regions,
}


# Stage -> the `OutputPaths` key of the input it reads. Checked before a run, so skipping the stage
# that writes it works only if it's already in storage.
NEEDS: dict[str, str] = {'pyramid': 'store', 'buildings': 'store', 'regions': 'buildings'}


def check_storage(
    stages: Sequence[str],
    spec: RiskSpec,
    out: OutputPaths,
    overwrite: bool,
    kwargs: Mapping[str, Mapping[str, Any]] = {},
) -> list[str]:
    """One message per requested stage whose input is neither produced in this run nor in storage,
    one per missing TIGER input (`pixi run regions-etl`), and one if etl would refuse an existing
    store (it raises only once the job is running). `kwargs`: per stage, the kwargs it was given.
    """
    cfg = StorageConfig()
    store = cfg.object_store()
    # `OutputPaths` key -> the stage that writes it. Without score (flood: FDP is the risk layer),
    # etl writes the store.
    writers = {'store': 'score' if spec.score else 'etl', 'buildings': 'buildings'}

    def exists(prefix: str) -> bool:
        # list matches whole path segments, so a file's own key lists nothing: head it first.
        try:
            obstore.head(store, prefix)
            return True
        except FileNotFoundError:
            return any(obstore.list(store, prefix=prefix, chunk_size=1))

    problems = []
    for s in stages:
        if s not in NEEDS:
            continue
        w, prefix = writers[NEEDS[s]], getattr(out, NEEDS[s])
        if w not in stages and not exists(prefix):
            problems.append(f'{s} needs {w} output at {cfg.uri(prefix)}: run {w} first')
    tiger: dict[str, Sequence[RegionLevel]] = {
        'buildings': [] if kwargs.get('buildings', {}).get('blocks') else [RegionLevel.BLOCK],
        'regions': kwargs.get('regions', {}).get('levels', tuple(RegionLevel)),
    }
    for s in (s for s in stages if s in tiger):
        for level in map(RegionLevel, tiger[s]):
            if not exists(level.key):
                problems.append(f'{s} needs TIGER {level} at {level.uri}: run pixi run regions-etl')
    if 'etl' in stages and writers['store'] == 'etl' and not overwrite and exists(out.store):
        problems.append(f'etl: {cfg.uri(out.store)} exists: pass --overwrite or skip etl')
    return problems


def unimplemented(stages: Sequence[str], spec: RiskSpec) -> list[str]:
    """One message per requested stage this risk has no implementation for."""
    # pyramid has no RiskSpec function; a store CRS marks it implemented (hurricane has none).
    needs = {'etl': spec.etl, 'score': spec.score, 'pyramid': spec.crs}
    return [
        f'{s}: not implemented for this risk' for s in stages if s in needs and needs[s] is None
    ]


# Stage -> the util it calls, then the utils that util passes its **kwargs to. A stage accepts
# their params that have defaults; the first util that names a param owns it.
STAGE_KWARGS: dict[str, tuple[Callable[..., Any], ...]] = {
    'pyramid': (write_multiscales, create_pyramid),
    'buildings': (write_buildings, write_buildings_lod),
    'regions': (write_regions,),
}


def stage_kwargs(stage: str, kw: Mapping[str, Any]) -> BaseModel:
    """Validate `kw` against `stage`'s utils (`STAGE_KWARGS`). Unknown keys and bad types raise."""
    fields: dict[str, Any] = {}
    for fn in STAGE_KWARGS.get(stage, ()):
        hints = get_type_hints(fn)
        for p in signature(fn).parameters.values():
            # overwrite is the stage's own flag (--overwrite); metadata and log are passed by the
            # calling util (write_buildings), so a user value would collide.
            if p.name in {'overwrite', 'metadata', 'log'}:
                continue
            if p.default is not p.empty and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD):
                fields.setdefault(p.name, (hints.get(p.name, Any), p.default))
    config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    return create_model(f'{stage}_kwargs', __config__=config, **fields).model_validate(kw)
