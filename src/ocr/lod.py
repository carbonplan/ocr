"""Building GeoParquet for analysis and the map in one file (Buildings from the work done in carbonplan/ocr-multi-hazard#8).

The output is GeoParquet with every source column and the original geometry, plus
geoparquet-lod overviews: per-zoom simplified copies, Hilbert order and a spatial index,
which a browser reads as map tiles without PMTiles.
"""

import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import obstore
from geoparquet_lod import chain, combine, density_budget, drop_smallest, only, tiny_polygons, write

from ocr.config import StorageConfig
from ocr.duck import duckdb_connect
from ocr.geoparquet import kv_metadata, write_relation

# Buildings kept in the busiest tile at each zoom, up to and including the key's zoom. One
# fraction is kept everywhere at a zoom, so relative density reads the same at every scale.
DEFAULT_TILE_BUDGETS: Mapping[int, int] = {
    5: 70_000,
    7: 50_000,
    10: 70_000,
    11: 120_000,
    12: 80_000,
    13: 120_000,
}
# Below this zoom the map draws only a uniform sample of the `shown` buildings, as a dot layer
# would; from it, every building is also thinned the way tippecanoe thins polygons (smallest
# dropped first, tiny ones folded into squares), so the biggest show first as the map zooms in.
POLYGONS_FROM_ZOOM = 6
# Estimated bytes of that polygon thinning in the busiest tile at each zoom.
POLYGON_TILE_BYTES = 850_000
# Every building is drawn from this zoom.
EVERY_FROM_ZOOM = 13
# Full detail from this zoom; the map overzooms it past.
COMPLETE_ZOOM = 14
# A building under a pixel draws as a square this many pixels wide.
PLACEHOLDER_PIXELS = 0.25


def display_thinning(
    shown: str,
    tile_budgets: Mapping[int, int] = DEFAULT_TILE_BUDGETS,
    polygons_from: int = POLYGONS_FROM_ZOOM,
    polygon_tile_bytes: int = POLYGON_TILE_BYTES,
    every_from: int = EVERY_FROM_ZOOM,
) -> Callable:
    """Rows each zoom draws: a uniform sample of rows matching the SQL predicate `shown`, sized so
    the busiest tile at the zoom holds its budget; from `polygons_from` also every row thinned
    smallest first; from `every_from` every row."""
    limits = sorted(tile_budgets.items())

    def budget(zoom: int) -> int:
        return next((n for top, n in limits if zoom <= top), limits[-1][1])

    sample = only(shown, density_budget(budget))
    with_polygons = combine(sample, chain(drop_smallest(polygon_tile_bytes), tiny_polygons()))

    def thin(rows, zoom, ctx):
        if zoom >= every_from:
            return rows
        return (with_polygons if zoom >= polygons_from else sample)(rows, zoom, ctx)

    return thin


def write_buildings_lod(
    src: str | Path,
    out_key: str,
    *,
    shown: str,
    carry: Sequence[str | tuple[str, str]],
    metadata: Mapping[str, str] | None = None,
    tile_budgets: Mapping[int, int] = DEFAULT_TILE_BUDGETS,
    polygons_from: int = POLYGONS_FROM_ZOOM,
    polygon_tile_bytes: int = POLYGON_TILE_BYTES,
    every_from: int = EVERY_FROM_ZOOM,
    complete_zoom: int = COMPLETE_ZOOM,
    placeholder_pixels: float = PLACEHOLDER_PIXELS,
    memory_limit: str | None = None,
    workers: int | None = None,
    temp_dir: str | Path | None = None,
    log: Callable = print,
) -> None:
    """Write the buildings in `src` (a GeoParquet local or s3://, e.g. a `sample_buildings` output
    or this function's own output) with overviews to storage key `out_key`, keeping its
    key/value metadata plus `metadata`.

    `shown` is a SQL predicate for the buildings coarse zooms draw (`RiskOutputs.lod_shown`);
    `carry` names the columns tiles carry for styling, each a name or a
    (name, Arrow type) pair such as ('rps_2011', 'halffloat'). `temp_dir` holds the local files (on
    Coiled, a large volume). The rest are the thinning knobs.
    """
    # TODO: geoparquet-lod's write() needs `src` as a parquet path: it reads it several times (schema,
    # geometry sample, validity counts, Hilbert sort pass), so `write_buildings` stages a temp file;
    # drop it if upstream accepts a re-runnable query or an Arrow RecordBatchReader it spills itself.
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp, duckdb_connect() as con:
        meta = kv_metadata(con, str(src)) | dict(metadata or {})
        rel = con.sql('SELECT * FROM read_parquet($p)', params={'p': str(src)})
        if 'lod' in rel.columns:
            # Our own output: the writer refuses its lod/hilbert columns, so rerun from the rest.
            src = Path(tmp) / 'src.parquet'
            write_relation(rel.select('* EXCLUDE (lod, hilbert)'), meta, src)
        out = Path(tmp) / 'buildings.parquet'
        write(
            str(src),
            out,
            thinning=display_thinning(
                shown, tile_budgets, polygons_from, polygon_tile_bytes, every_from
            ),
            carry=list(carry),
            zooms=list(range(complete_zoom + 1)),
            complete_from=complete_zoom,
            placeholder_pixels=placeholder_pixels,
            metadata=meta,
            memory_limit=memory_limit,
            workers=workers,
            temp_dir=tmp,
            log=log,
        )
        log(f'uploading {out} to {out_key}')
        obstore.put(StorageConfig().object_store(), out_key, out)
