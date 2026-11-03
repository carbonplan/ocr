"""Census TIGER/Line 2026 region polygons for ocr.regions.stats: CONUS, full resolution (tl, not cb).

Writes RegionLevel.uri per level with GEOID, geometry, bbox (+ NAME on states/counties, STUSPS
on states). Only what regions.stats reads. Parent codes are left(GEOID, n) at every level: block
files key GEOID on 2020 codes, so the block GEOID is rebuilt from the current STATEFP/COUNTYFP/
TRACTCE (CT moved to planning-region counties in 2022; without this its blocks match no county).
Nation is the dissolve of the states, so it needs states written first.

Run: pixi run regions-etl [--level states --level nation] [--overwrite]
Blocks download 49 zips (~8 GB of geometry); run them on a VM:
coiled run --vm-type c8g.8xlarge --region us-west-2 pixi run regions-etl --level blocks
"""

import tempfile
from collections.abc import Iterator
from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import NamedTuple

import duckdb
import obstore
import pyarrow as pa
import typer
from obstore.store import HTTPStore
from upath import UPath

from ocr.config import StorageConfig
from ocr.duck import duckdb_connect
from ocr.geoparquet import bbox_sql, write_geoparquet
from ocr.regions import TIGER_VINTAGE, RegionLevel

TIGER = f'https://www2.census.gov/geo/tiger/TIGER{TIGER_VINTAGE}'
# CONUS states + DC, copied from fire's census_tiger.FIPS_CODES (importing it pulls in coiled + geopandas).
CONUS_FIPS = (
    '01 04 05 06 08 09 10 11 12 13 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 '
    '31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56'
).split()
# The national state/county files include AK, HI, PR and the territories; this keeps CONUS.
# repr() adds the SQL quotes: "WHERE STATEFP IN ('01', '04', ..., '56')".
IN_CONUS = f'WHERE STATEFP IN ({", ".join(repr(f) for f in CONUS_FIPS)})'
# Current-geography block key, e.g. CT 09013 5352 003010 -> 09110 5352 003010. Two CT water
# blocks from old counties 007 and 011 land in one new tract with the same code, hence merge_dupes.
BLOCK_GEOID = "STATEFP || COUNTYFP || TRACTCE || BLOCKCE20 || coalesce(SUFFIXCE, '') AS GEOID"


class Source(NamedTuple):
    """TIGER zips (paths under TIGER) for one level, the columns to keep, an optional row filter, and whether to
    union polygons that share a GEOID."""

    paths: list[str]
    columns: str
    where: str = ''
    merge_dupes: bool = False


V = TIGER_VINTAGE
SOURCES = {
    RegionLevel.STATE: Source([f'STATE/tl_{V}_us_state.zip'], 'GEOID, STUSPS, NAME', IN_CONUS),
    RegionLevel.COUNTY: Source([f'COUNTY/tl_{V}_us_county.zip'], 'GEOID, NAME', IN_CONUS),
    RegionLevel.TRACT: Source([f'TRACT/tl_{V}_{f}_tract.zip' for f in CONUS_FIPS], 'GEOID'),
    RegionLevel.BLOCK: Source(
        [f'TABBLOCKSUFX/tl_{V}_{f}_tabblocksufx.zip' for f in CONUS_FIPS],
        BLOCK_GEOID,
        merge_dupes=True,
    ),
}


# obstore retries failed requests with backoff (default RetryConfig), so a flaky census.gov response
# doesn't kill a 49-zip run. Its default 30 s timeout covers the whole body; big state zips need longer.
STORE = HTTPStore.from_url(TIGER, client_options={'timeout': timedelta(minutes=30)})


def download(path: str, dest: Path) -> None:
    """Stream TIGER `path` (relative to TIGER) to `dest`."""
    with dest.open('wb') as f:
        for chunk in obstore.get(STORE, path).stream():
            f.write(chunk)


def read_zips(con: duckdb.DuckDBPyConnection, src: Source, tmp: Path) -> pa.RecordBatchReader:
    """Stream all the zips' shapefiles as one table, one zip at a time: download, read, delete.

    Peak disk and memory is one state, not all 49 (a UNION ALL of every zip ran a laptop out of
    disk). The reader pulls rows as the writer asks for them. `tmp` must outlive the reader.
    """

    def batches() -> Iterator[pa.RecordBatch]:
        for path in src.paths:
            zip_path = tmp / Path(path).name
            download(path, zip_path)
            # /vsizip/ lets GDAL (inside DuckDB spatial) read the .shp inside the zip without
            # unzipping, e.g. ST_Read('/vsizip//tmp/x/tl_2026_06_tract.zip'). Its geometry is `geom`.
            q = f"SELECT {src.columns}, geom AS geometry FROM ST_Read('/vsizip/{zip_path}') {src.where}"
            if src.merge_dupes:
                q = f'SELECT GEOID, ST_Union_Agg(geometry) AS geometry FROM ({q}) GROUP BY GEOID'
            yield from con.sql(
                f'SELECT *, {bbox_sql("geometry")} AS bbox FROM ({q})'
            ).to_arrow_reader()
            zip_path.unlink()

    # The writer needs the schema up front, so read the first batch to get it.
    it = batches()
    first = next(it)
    return pa.RecordBatchReader.from_batches(first.schema, chain([first], it))


def nation_sql(states: str) -> str:
    """One CONUS outline: ST_Union_Agg dissolves the state polygons, so interior borders vanish.

    Great Lakes and coastal water stay in the outline, since tl states include them.
    """
    return f"""
        SELECT *, {bbox_sql('geometry')} AS bbox
        FROM (SELECT 'US' AS GEOID, 'United States' AS NAME, ST_Union_Agg(geometry) AS geometry
              FROM read_parquet('{states}'))
    """


def exists(key: str) -> bool:
    try:
        obstore.head(StorageConfig().object_store(), key)
    except FileNotFoundError:
        return False
    return True


def main(
    level: list[RegionLevel] = typer.Option(list(RegionLevel), '--level'),
    overwrite: bool = typer.Option(False, '--overwrite'),
) -> None:
    # RegionLevel lists NATION first, but nation is dissolved from the states written here,
    # so go finest to coarsest.
    for lvl in [lvl for lvl in reversed(RegionLevel) if lvl in level]:
        if exists(lvl.key) and not overwrite:
            typer.echo(f'skip {lvl}: {lvl.uri} exists; pass --overwrite')
            continue
        # s3_write adds AWS creds, which nation needs to read the states back from S3.
        with (
            tempfile.TemporaryDirectory() as tmp,
            duckdb_connect(s3_write=UPath(lvl.uri).protocol == 's3') as con,
        ):
            if lvl is RegionLevel.NATION:
                rows = con.sql(nation_sql(RegionLevel.STATE.uri)).to_arrow_reader()
            else:
                rows = read_zips(con, SOURCES[lvl], Path(tmp))
            write_geoparquet(rows, lvl.uri)
        typer.echo(f'wrote {lvl} to {lvl.uri}')


if __name__ == '__main__':
    typer.run(main)
