import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from ocr.config import TIGER_VINTAGE
from ocr.duck import duckdb_connect
from ocr.geoparquet import kv_metadata, write_relation
from ocr.regions import RegionLevel


@dataclass(frozen=True)
class Hist:
    """Variables that share one set of histogram bin edges."""

    variables: Sequence[str]
    bins: Sequence[float]

    def __post_init__(self) -> None:
        # Plain tuples of floats: numpy bins break `not self.bins` and repr as np.float64(...) in SQL.
        object.__setattr__(self, 'variables', tuple(self.variables))
        object.__setattr__(self, 'bins', tuple(float(b) for b in self.bins))
        if not self.bins or any(lo >= hi for lo, hi in pairwise(self.bins)):
            raise ValueError(f'bins must be non-empty and strictly ascending: {self.bins}')
        if not all(math.isfinite(b) and b > 0 for b in self.bins):
            raise ValueError(f'bins must be finite and > 0: {self.bins}')


# Building columns that are not values to summarise: ids, geometry and geoparquet-lod's map columns.
NOT_VALUES = ('id', 'GEOID', 'geometry', 'bbox', 'lod', 'hilbert')


def _hist_sql(var: str, bins: Sequence[float]) -> str:
    """SQL for the hist of `var`: [x = 0, x <= bins[0], bins[0] < x <= bins[1], ..., x > bins[-2]].

    The hist covers every non-NULL value, so it sums to the count mean and median use: negatives fall
    in the first nonzero bin (as in fire) and values above the last edge in the last bin (fire dropped them).
    Example, bins=(0.01, 0.1): hist = [x = 0, x != 0 AND x <= 0.01, x > 0.01]; -1 is in bin 1, 0.5 in bin 2.
    So bins[-1] never cuts: it is the last bin's display cap, kept for fire's 11-entry shape.
    """
    x = f'"{var}"'
    # Last bin is open (b[-2], inf), so every hist sums to the count; fire capped it at b[-1] and dropped the rest.
    # To revisit: swap "'inf'" for repr(bins[-1]) to cap like fire, or drop bins[-1] if no reader needs the label.
    cuts = ["'-inf'", *(repr(b) for b in bins[:-1]), "'inf'"]
    edges = [f'{x} = 0'] + [
        f'{x} != 0 AND {x} > {lo}::DOUBLE AND {x} <= {hi}::DOUBLE' for lo, hi in pairwise(cuts)
    ]
    counts = ', '.join(f'count(*) FILTER ({e})' for e in edges)
    return f'[{counts}] AS "{var}_hist"'


def write_region_stats(
    buildings: str | Path,
    level: RegionLevel,
    hists: Sequence[Hist],
    out: str | Path,
    regions: str | Path | None = None,
) -> None:
    """Write to `out` per-region building_count, mean, median and missing count of every building value
    column, hist of each `hists` var, plus the region's columns.

    `buildings` is a GeoParquet with GEOID (`write_buildings` output); its value columns are all but
    NOT_VALUES. Each building rolls up from its block GEOID by prefix (`level.group_key`), so levels
    nest and a building counts once per level, the same lookup the web does. `regions` (default
    `level.uri`) needs GEOID, geometry and bbox; its other columns (NAME, STUSPS) are kept. Buildings
    with no block and regions with no buildings are dropped. NaN counts as missing. Metadata: the
    buildings' own, plus bins and TIGER vintage.
    """
    hist_vars = [v for h in hists for v in h.variables]
    if len(set(hist_vars)) != len(hist_vars):
        raise ValueError(f'variable in more than one Hist: {hist_vars}')
    params = {'buildings': str(buildings), 'regions': str(regions or level.uri)}
    with duckdb_connect() as con:
        meta = kv_metadata(con, params['buildings'])
        columns = con.sql(
            'SELECT * FROM read_parquet($b) LIMIT 0', params={'b': params['buildings']}
        ).columns
        variables = [c for c in columns if c not in NOT_VALUES]
        if missing := set(hist_vars) - set(variables):
            raise ValueError(f'Hist variables not in buildings: {sorted(missing)}')
        values = ''.join(f', nullif("{v}"::DOUBLE, \'NaN\'::DOUBLE) AS "{v}"' for v in variables)
        stats = ['count(*) AS building_count']
        stats += [
            f'avg("{v}") AS "{v}_mean", median("{v}") AS "{v}_median", '
            f'count(*) FILTER ("{v}" IS NULL) AS "{v}_missing"'
            for v in variables
        ]
        stats += [_hist_sql(v, h.bins) for h in hists for v in h.variables]
        rel = con.sql(
            f"""
            -- each building's GEOID at this level; NaN -> NULL
            -- can_have_nan: fallback for files with NaN; ours write NULL (buildings._sample, duckdb#25521)
            WITH tagged AS (
                SELECT {level.group_key} AS GEOID{values}
                FROM read_parquet($buildings, can_have_nan = true)
                WHERE GEOID IS NOT NULL
            ),
            stats AS (
                SELECT GEOID, {', '.join(stats)}
                FROM tagged
                GROUP BY 1
            )
            -- attach the region's columns, geometry last as GeoParquet readers expect
            SELECT s.GEOID, r.* EXCLUDE (GEOID, geometry, bbox), s.* EXCLUDE (GEOID), r.geometry, r.bbox
            FROM stats s
            JOIN read_parquet($regions) r ON s.GEOID = r.GEOID
            ORDER BY s.GEOID
            """,
            params=params,
        )
        bins = {v: list(h.bins) for h in hists for v in h.variables}
        write_relation(rel, meta | {'bins': json.dumps(bins), 'tiger_vintage': TIGER_VINTAGE}, out)
