"""Per-risk output settings and the shared stages: score, pyramid, buildings, regions."""

import os
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
from ocr.multiscales import drop_levels, write_multiscales
from ocr.regions import RegionLevel
from ocr.regions.layout import write_level
from ocr.regions.stats import Hist, write_region_stats
from ocr.risks.fire import score as fire_score

# Fire rps hist, ported from the retired fire_wind_risk_regional_aggregator (it had no bp hists).
FIRE_RPS_BINS = (0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.5, 1, 3, 100)
# ponytail: rps bins / 100 (rps % = bp x crps %, crps up to 100); bp is a 0-1 fraction. Refit from
# a CONUS quantile pass (in-region Coiled). Too low: 5 regions (2026-11-03) put ~90% of bp > 0 above
# 0.001 (bp_2011 p50 0.004, p99 0.046; bp_2047 p50 0.006, p99 0.105). Candidate from those:
# (0.0005, 0.001, 0.0015, 0.0025, 0.004, 0.006, 0.01, 0.02, 0.05, 1).
FIRE_BP_BINS = tuple(b / 100 for b in FIRE_RPS_BINS)
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
    # None -> the risk's own ETL writes store.icechunk group "0" (flood: FDP is the risk layer).
    # (out, part): writes one part (fire: a region id) into store.icechunk group "0".
    score: Callable[[OutputPaths, str], None] | None = None
    # (out, **kwargs) -> parts not written yet; creates the store if missing. Set with score.
    score_parts: Callable[..., list[str]] | None = None


RISKS = {
    Risk.FIRE: RiskSpec(
        hists=[
            Hist(['rps_2011', 'rps_2047'], FIRE_RPS_BINS),
            Hist(['bp_2011', 'bp_2047'], FIRE_BP_BINS),
        ],
        lod_shown=positive('rps_2011', 'rps_2047'),
        # half precision: the map colours by bins; selecting a building reads full precision by row
        lod_carry=[(v, 'halffloat') for v in ('rps_2011', 'rps_2047', 'bp_2011', 'bp_2047')],
        crs='EPSG:4326',
        score=fire_score.write_region,
        score_parts=fire_score.parts,
    ),
    # ponytail: even 0.1 bins from six city windows; revisit with a CONUS quantile pass (in-region Coiled).
    Risk.FLOOD: RiskSpec(
        hists=[Hist(['fdp'], FLOOD_FDP_BINS)],
        lod_shown=positive('fdp'),
        lod_carry=[('fdp', 'halffloat')],
        crs='EPSG:5070',
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


def _exists(store: obstore.store.ObjectStore, key: str) -> bool:
    # list matches whole path segments, so a file's own key lists nothing: head it first.
    try:
        obstore.head(store, key)
        return True
    except FileNotFoundError:
        return any(obstore.list(store, prefix=key, chunk_size=1))


def _delete_prefix(prefix: str, chunk: int = 20_000) -> None:
    """Delete every object under `prefix`. obstore sends S3 DeleteObjects of 1000 keys, 20 at a
    time, and retries (`object_store` config); chunks only give progress output.
    """
    store = StorageConfig().object_store()
    paths = [m['path'] for batch in obstore.list(store, prefix=prefix) for m in batch]
    for i in range(0, len(paths), chunk):
        obstore.delete(store, paths[i : i + chunk])
        print(f'\rdeleting {prefix}: {min(i + chunk, len(paths))}/{len(paths)}', end='', flush=True)
    if paths:
        print()


# Stages take (spec, out, overwrite, **util kwargs). Storage is the hand-off: each reads what an
# earlier one wrote (`preflight`'s `reads`).

# Set by Coiled in each task of a map job (`cli.run_score`): the one part that task writes.
TASK_INPUT = 'COILED_BATCH_TASK_INPUT'


def plan_score(spec: RiskSpec, out: OutputPaths, overwrite: bool, **kw: Any) -> list[str]:
    """Parts not written yet (`spec.score_parts`); overwrite deletes the store first.
    Parts to write drop the pyramid levels first, so they never go stale over group "0" (as
    flood-etl --overwrite); rerun `pyramid`. Only the driver plans, so tasks never drop.
    """
    assert spec.score_parts is not None  # `preflight`
    if overwrite and kw.get('region_ids') is not None:
        # Overwrite deletes the whole store; a subset would leave every other region NaN.
        raise ValueError('--overwrite deletes every region; drop region_ids or --overwrite')
    if overwrite:
        _delete_prefix(out.store)
    if todo := spec.score_parts(out, **kw):
        _drop_pyramid(out)
    return todo


def _drop_pyramid(out: OutputPaths) -> None:
    session = _store(out).writable_session('main')
    drop_levels(session)
    if session.has_uncommitted_changes:  # no levels: no commit
        session.commit('drop pyramid levels: score writes new parts')


def score(spec: RiskSpec, out: OutputPaths, overwrite: bool, **kw: Any) -> None:
    # A Coiled task writes its one part; the driver planned (and overwrote) already.
    assert spec.score is not None
    if part := os.environ.get(TASK_INPUT):
        spec.score(out, part)
        return
    # In process: plan, write every part. A failure raises; rerunning resumes from the commits.
    for part in plan_score(spec, out, overwrite, **kw):
        spec.score(out, part)


def _store(out: OutputPaths) -> icechunk.Repository:
    return icechunk.Repository.open(StorageConfig().icechunk_storage(out.store))


def pyramid(spec: RiskSpec, out: OutputPaths, overwrite: bool, **kw: Any) -> None:
    # A rerun with the same settings replaces the levels; overwrite drops them first, base kept.
    assert spec.crs is not None
    if 'levels' not in kw and 'factors' not in kw:
        kw['levels'] = 'auto'  # create_pyramid requires one of them
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


# Run order.
STAGES: dict[str, Callable[..., None]] = {
    'score': score,
    'pyramid': pyramid,
    'buildings': buildings,
    'regions': regions,
}


def preflight(
    stages: Sequence[str],
    spec: RiskSpec,
    out: OutputPaths,
    kwargs: Mapping[str, Mapping[str, Any]] = {},
) -> list[str]:
    """One message per problem that would otherwise surface only once a stage is running: no code
    for this risk, input neither written in this run nor in storage, or TIGER missing.
    `kwargs`: per stage, the kwargs it was given.
    """
    cfg = StorageConfig()
    store = cfg.object_store()
    store_writer = 'score' if spec.score else f'the {out.risk} ETL'
    implemented = {'score': spec.score, 'pyramid': spec.crs}
    reads = {  # stage -> (input key, what writes it)
        'pyramid': (out.store, store_writer),
        'buildings': (out.store, store_writer),
        'regions': (out.buildings, 'buildings'),
    }
    tiger = {
        'buildings': [] if kwargs.get('buildings', {}).get('blocks') else [RegionLevel.BLOCK],
        'regions': kwargs.get('regions', {}).get('levels', tuple(RegionLevel)),
    }
    problems = []
    for s in stages:
        if s in implemented and implemented[s] is None:
            problems.append(f'{s}: not implemented for {out.risk}')
            continue
        if s in reads:
            key, writer = reads[s]
            if writer not in stages and not _exists(store, key):
                problems.append(f'{s} needs {writer} output at {cfg.uri(key)}: run {writer} first')
        problems += [
            f'{s} needs TIGER {level} at {level.uri}: run pixi run regions-etl'
            for level in map(RegionLevel, tiger.get(s, ()))
            if not _exists(store, level.key)
        ]
    return problems


# Stage -> the util it calls, then the utils that util passes its **kwargs to. A stage accepts
# their params that have defaults; the first util that names a param owns it.
STAGE_KWARGS: dict[str, tuple[Callable[..., Any], ...]] = {
    # Fire is the only risk with score; a second one needs per-risk kwargs here.
    'score': (fire_score.parts,),
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
