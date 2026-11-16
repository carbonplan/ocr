"""IBTrACS tropical cyclone tracks (NOAA NCEI, v04r01) from the 2004 season on, for storms whose winds reached
CONUS, as GeoParquet.

One row per 3-hourly best-track segment (a centerline), every segment of each storm, weak stages included,
with its recorded wind and category. Segments with tropical-storm-force winds (USA_WIND >= 34 kt) carry
`r34_nmi`, the farthest extent of 34 kt winds, taken as the largest of NHC's four quadrant radii. A point
within `r34_nmi` (great-circle) of a segment was, by the best track, within reach of that storm's
tropical-storm-force winds; the web picks the storms at a point that way and decides which matter, e.g. by
`usa_wind` or `usa_sshs`. Candidates are the rows whose geometry lies within `ibtracs_max_r34_nmi` (file
metadata) of the point. A storm is kept when any segment, grown by its R34, touches CONUS.

Storms start at FIRST_SEASON, the season NHC best tracks began carrying wind radii routinely. Most earlier
segments lack them, so their R34 would be a typical size for their category, not a record of where their
winds reached, and the storms matched at a point would rest on guesses.

The file has no overviews. At ~270 KB it is read whole once by the geoparquet-lod reader, which draws every
zoom from the segments themselves. Rows are in one row group, sorted by storm and time, so each storm's
segments sit together and its attributes compress to almost nothing.

A segment with tropical-storm-force winds but no radii takes the radius recorded closest in time on the
same storm, within NEAREST_HOURS, the larger on a tie; `r34_source` is `observed` or `nearest`. NHC records
radii at its 6-hourly analyses while a storm is at least 34 kt, and IBTrACS interpolates them to its other
points, so a gap falls between an analysis with radii and one below 34 kt. Such gaps are rare (10 segments
in the 2026-11-08 source), nearly all at or just after a landfall as the storm weakens, where the last
radius likely overstates reach a little. A gap with no radius that near keeps no R34 and matches no point,
unless the largest radius in the source would carry it to CONUS, where it could change which storms reached
a point: then it fails the build.
Radii describe the largest quadrant, so they bound where those winds reached.

writes input/historic/ibtracs/<YYYYMMDD>/ibtracs_tracks.parquet, dated by the source's Last-Modified. The
downloaded zip is mirrored beside it at raw/<sha256[:16]>/<zip name>, content-addressed and never overwritten.

Cite Knapp et al. (2010), doi:10.1175/2009BAMS2755.1, and Gahtan et al. (2024), doi:10.25921/82ty-9e16.

run: pixi run historic-ibtracs [--overwrite]
"""

import hashlib
import tempfile
import urllib.request
from collections.abc import Mapping
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import duckdb
import obstore
import typer

from ocr.buildings import CONUS
from ocr.config import StorageConfig
from ocr.duck import duckdb_connect
from ocr.geoparquet import bbox_sql, kv_metadata, write_relation

ZIP_NAME = 'IBTrACS.since1980.list.v04r01.lines.zip'
URL = (
    'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/'
    f'v04r01/access/shapefile/{ZIP_NAME}'
)
SHAPEFILE = 'IBTrACS.since1980.list.v04r01.lines.shp'
ATTRIBUTION = 'IBTrACS v04r01 (NOAA NCEI)'
LICENSE = 'Full and open access (World Data Center policy); commercial use per WMO Resolution 40'
TS_WIND_KT = 34
NMI_TO_M = 1852.0
# The shortest degree of latitude and the equatorial degree of longitude, so R34 in degrees never falls short.
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQUATOR = 111_320.0
FIRST_SEASON = 2004
# Hours on the same storm within which a segment without radii takes the nearest recorded one.
NEAREST_HOURS = 6
ZSTD_LEVEL = 9
# pyproject.toml [tool.setuptools_scm] fallback_version: what a build without repository metadata reports.
SCM_FALLBACK_VERSION = '999'


def key_for(day: date) -> str:
    return f'input/historic/ibtracs/{day:%Y%m%d}/ibtracs_tracks.parquet'


def raw_key(day: date, sha256: str) -> str:
    """Content-addressed, so two different zips published on one date get two mirrors, never one replaced."""
    return f'input/historic/ibtracs/{day:%Y%m%d}/raw/{sha256[:16]}/{ZIP_NAME}'


def source_date(last_modified: str | None, *, today: date | None = None) -> date:
    """The HTTP Last-Modified header's date (in UTC), or `today` when the header is missing."""
    if not last_modified:
        return today or datetime.now(UTC).date()
    return parsedate_to_datetime(last_modified).astimezone(UTC).date()


def source_metadata(
    url: str, *, last_modified: str | None, sha256: str, retrieved: date, raw_mirror: str
) -> dict[str, str]:
    return {
        'ibtracs_url': url,
        'ibtracs_last_modified': last_modified or '',
        'ibtracs_sha256': sha256,
        'ibtracs_raw_mirror': raw_mirror,
        'ibtracs_retrieved': retrieved.isoformat(),
        'ibtracs_attribution': ATTRIBUTION,
        'ibtracs_license': LICENSE,
    }


def ocr_version() -> str:
    version = importlib_metadata.version('ocr')
    return f'unknown ({version})' if version == SCM_FALLBACK_VERSION else version


def processing_metadata() -> dict[str, str]:
    """The settings and code versions that shaped the output, so a file is self-describing."""
    return {
        'ibtracs_ocr_version': ocr_version(),
        'ibtracs_first_season': str(FIRST_SEASON),
        'ibtracs_min_wind_kt': str(TS_WIND_KT),
        'ibtracs_nearest_hours': str(NEAREST_HOURS),
    }


def shapefile_source(zip_path: str | Path) -> str:
    """SQL table expression reading the track-segment shapefile inside the zip (geometry column `geom`)."""
    return f"ST_Read('/vsizip/{zip_path}/{SHAPEFILE}')"


def reaches_conus_sql(geom: str, r34_nmi: str) -> str:
    """SQL true where the bbox of `geom` grown by `r34_nmi` on every side touches CONUS."""
    dlat = f'({r34_nmi} * {NMI_TO_M} / {M_PER_DEG_LAT})'
    ymin = f'greatest(ST_YMin({geom}) - {dlat}, -90)'
    ymax = f'least(ST_YMax({geom}) + {dlat}, 90)'
    # Longitude degrees are shortest at the most poleward latitude the box reaches.
    poleward = f'least(greatest(abs({ymin}), abs({ymax})), 89)'
    dlon = f'({r34_nmi} * {NMI_TO_M} / ({M_PER_DEG_LON_EQUATOR} * cos(radians({poleward}))))'
    return (
        f'ST_XMin({geom}) - {dlon} <= {CONUS.right} AND ST_XMax({geom}) + {dlon} >= {CONUS.left} '
        f'AND {ymin} <= {CONUS.top} AND {ymax} >= {CONUS.bottom}'
    )


def tracks(con: duckdb.DuckDBPyConnection, source: str) -> duckdb.DuckDBPyRelation:
    """Every segment of the storms since FIRST_SEASON whose winds reached CONUS from `source`, a table or table expression with
    the IBTrACS columns and `geom` in EPSG:4326, with R34 (see the module docstring)."""
    strong = f'usa_wind >= {TS_WIND_KT}'
    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE ibtracs_segments AS
        WITH segs AS (
            SELECT row_number() OVER () AS seg, SID AS sid, NAME AS name, SEASON AS season,
                   BASIN AS basin, ISO_TIME AS iso_time, NATURE AS nature, USA_WIND AS usa_wind,
                   USA_SSHS AS usa_sshs,
                   greatest(USA_R34_NE, USA_R34_SE, USA_R34_SW, USA_R34_NW) AS r34_observed,
                   epoch(CAST(ISO_TIME AS TIMESTAMP)) AS t, geom
            FROM {source}
            WHERE SEASON >= {FIRST_SEASON}
        ),
        nearest AS (
            SELECT gap.seg,
                   arg_min(known.r34_observed, [abs(known.t - gap.t), -known.r34_observed]) AS r34_nearest
            FROM segs gap
            JOIN segs known
              ON known.sid = gap.sid AND known.r34_observed > 0
             AND abs(known.t - gap.t) <= {NEAREST_HOURS} * 3600
            WHERE gap.{strong} AND NOT coalesce(gap.r34_observed > 0, false)
            GROUP BY gap.seg
        ),
        sized AS (
            SELECT segs.* EXCLUDE (seg, r34_observed, t, geom),
                   CAST(CASE WHEN NOT {strong} THEN NULL
                             WHEN r34_observed > 0 THEN r34_observed ELSE r34_nearest END AS DOUBLE)
                       AS r34_nmi,
                   CASE WHEN NOT {strong} THEN NULL
                        WHEN r34_observed > 0 THEN 'observed'
                        WHEN r34_nearest IS NOT NULL THEN 'nearest' END AS r34_source,
                   geom AS geometry
            FROM segs LEFT JOIN nearest USING (seg)
        )
        SELECT *, {bbox_sql('geometry')} AS bbox FROM sized
        """
    )
    # Checked before storms are picked, since a segment without R34 cannot reach CONUS and could drop its storm.
    # Any gap could matter when the source records no radius at all.
    largest = con.sql('SELECT max(r34_nmi) FROM ibtracs_segments').fetchall()[0][0]
    could_reach = reaches_conus_sql('geometry', str(largest)) if largest else 'true'
    missing = con.sql(
        'SELECT sid, iso_time FROM ibtracs_segments '
        f'WHERE {strong} AND r34_nmi IS NULL AND {could_reach} ORDER BY ALL'
    ).fetchall()
    if missing:
        raise ValueError(
            f'no R34 within {NEAREST_HOURS} h on the same storm near CONUS for (sid, iso_time) {missing}'
        )
    return con.sql(
        f"""
        SELECT * FROM ibtracs_segments
        WHERE sid IN (
            SELECT sid FROM ibtracs_segments WHERE {reaches_conus_sql('geometry', 'r34_nmi')}
        )
        """
    )


def write_local(
    zip_path: str | Path, out_dir: str | Path, *, metadata: Mapping[str, str] | None = None
) -> Path:
    """Write the tracks from the IBTrACS zip as GeoParquet to out_dir/ibtracs_tracks.parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'ibtracs_tracks.parquet'
    with duckdb_connect() as con:
        rel = tracks(con, shapefile_source(zip_path))
        max_r34 = rel.aggregate('max(r34_nmi)').fetchall()[0][0]
        meta = processing_metadata() | {'ibtracs_max_r34_nmi': str(max_r34)} | dict(metadata or {})
        write_relation(rel.order('sid, iso_time'), meta, out, zstd_level=ZSTD_LEVEL)
    return out


def download(url: str, dest: Path) -> tuple[str, str | None]:
    """Stream `url` to `dest`; returns its sha256 and Last-Modified header (None if absent)."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=300) as resp, dest.open('wb') as f:
        last_modified = resp.headers.get('Last-Modified')
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            f.write(chunk)
    return digest.hexdigest(), last_modified


def exists(key: str) -> bool:
    try:
        obstore.head(StorageConfig().object_store(), key)
    except FileNotFoundError:
        return False
    return True


def mirror(zip_path: Path, key: str) -> None:
    """Upload the raw zip to `key` unless it is already there. Mirrors are never overwritten."""
    if not exists(key):
        obstore.put(StorageConfig().object_store(), key, zip_path)
        typer.echo(f'mirrored {key}')


def built_from_sha256(key: str) -> str | None:
    """The source zip's sha256 recorded in the output at `key` (reads only the parquet footer)."""
    with duckdb_connect() as con:
        return kv_metadata(con, StorageConfig().uri(key)).get('ibtracs_sha256')


def main(overwrite: bool = typer.Option(False, '--overwrite')) -> None:
    # TMPDIR, else the system temp dir; on Coiled pass --env TMPDIR=/scratch (VM disk, not overlay /tmp).
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / ZIP_NAME
        sha256, last_modified = download(URL, zip_path)
        day = source_date(last_modified)
        key, raw = key_for(day), raw_key(day, sha256)
        if exists(key) and not overwrite:
            # The output must point at the bytes it was built from, so backfill only from the same zip.
            if built_from_sha256(key) == sha256:
                mirror(zip_path, raw)
            else:
                typer.echo(
                    f'{raw} not mirrored: the upstream file changed since {key} was built; '
                    'pass --overwrite to rebuild both'
                )
            typer.echo(f'skip: {key} exists; pass --overwrite')
            return
        meta = source_metadata(
            URL,
            last_modified=last_modified,
            sha256=sha256,
            retrieved=datetime.now(UTC).date(),
            raw_mirror=raw,
        )
        out = write_local(zip_path, Path(tmp) / 'out', metadata=meta)
        # After the build, so a failed one leaves no mirror behind.
        mirror(zip_path, raw)
        obstore.put(StorageConfig().object_store(), key, out)
    typer.echo(f'wrote {key}')


if __name__ == '__main__':
    typer.run(main)
