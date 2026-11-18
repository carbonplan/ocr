"""NOAA Storm Events Database (NCEI) flood reports, CONUS, as GeoParquet with overviews.

one point per Flood or Flash Flood report, at its begin location. NWS Storm Data gives a report two locations,
the corners bounding the flooded area (NWSI 10-1605, section 2.4), each referenced to the nearest town's center
and often to two decimals; `end_lon` and `end_lat` keep the second corner. a report is kept when its begin
location is inside the CONUS box, so a few placed just across the Canadian or Mexican border are included.
reports without a location are dropped, which removes nearly all of 1996 to 2004. Coastal Flood, Lakeshore Flood
and Storm Surge/Tide reports are located by forecast zone only and are not read. `event_id` and `episode_id` are
NCEI's. `damage` is property plus crop dollars, null when neither amount was entered. `deaths` and `injuries`
sum direct and indirect. `begin_date` and `end_date` are `YYYY-MM-DD` text, and `year` is the begin date's

writes input/historic/storm_events/<YYYYMMDD>/storm_events_floods.parquet. NCEI publishes one details file a
year and reissues it under a new creation stamp as reports are revised; the build reads each year's newest
file, and the key carries the newest stamp among them. each file is mirrored at
input/historic/storm_events/raw/<its name> (the stamp keeps a name's content fixed; uploaded after the build
succeeds), and a build reads a mirrored file from there

run: pixi run historic-storm-events [--overwrite]
"""

import hashlib
import json
import re
import tempfile
import urllib.request
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import duckdb
import geoparquet_lod
import obstore
import typer

from ocr.buildings import CONUS
from ocr.config import StorageConfig
from ocr.duck import duckdb_connect
from ocr.geoparquet import bbox_sql, write_relation

INDEX = 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'
# A details file's year, then its creation stamp.
DETAILS = re.compile(r'StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz')
# Flood and Flash Flood reports are recorded from this year.
FIRST_YEAR = 1996
EVENT_TYPES = ('Flood', 'Flash Flood')
ATTRIBUTION = 'NOAA NCEI Storm Events Database'
LICENSE = 'Public domain (US Government work)'
PREFIX = 'input/historic/storm_events'
RAW_PREFIX = f'{PREFIX}/raw'
ZOOMS = list(range(11))
THINNING = geoparquet_lod.density_budget()
# Fields copied into the overviews for styling and filtering; the web reads the rest from the rows.
CARRY = ['year', 'type']
SCM_FALLBACK_VERSION = '999'


def latest_files(index: str) -> dict[int, str]:
    """Each year's newest details file, from FIRST_YEAR, among those the NCEI directory listing `index` names."""
    latest: dict[int, str] = {}
    for match in DETAILS.finditer(index):
        year = int(match[1])
        # One year's names differ only in their stamp, so the greater name is the newer file.
        if year >= FIRST_YEAR and match[0] > latest.get(year, ''):
            latest[year] = match[0]
    return dict(sorted(latest.items()))


def source_date(names: Iterable[str]) -> date:
    """The newest creation stamp among the details files `names`."""
    return datetime.strptime(max(DETAILS.fullmatch(name)[2] for name in names), '%Y%m%d').date()


def key_for(day: date) -> str:
    return f'{PREFIX}/{day:%Y%m%d}/storm_events_floods.parquet'


def raw_key(name: str) -> str:
    return f'{RAW_PREFIX}/{name}'


def source_metadata(sha256: Mapping[str, str], *, retrieved: date) -> dict[str, str]:
    """`sha256` maps each details file the build read to its digest."""
    return {
        'storm_events_index': INDEX,
        'storm_events_files': json.dumps(dict(sorted(sha256.items()))),
        'storm_events_raw_mirror': f'{RAW_PREFIX}/',
        'storm_events_retrieved': retrieved.isoformat(),
        'storm_events_attribution': ATTRIBUTION,
        'storm_events_license': LICENSE,
    }


def ocr_version() -> str:
    """The ocr version, whose commit also pins geoparquet-lod in pyproject.toml."""
    version = importlib_metadata.version('ocr')
    return f'unknown ({version})' if version == SCM_FALLBACK_VERSION else version


def processing_metadata() -> dict[str, str]:
    """What shaped the output that geoparquet-lod doesn't already record (it writes zooms, carried fields and thinning)."""
    return {'storm_events_ocr_version': ocr_version()}


def date_sql(prefix: str) -> str:
    """SQL for the date in `prefix`_YEARMONTH ('201905') and `prefix`_DAY ('9'). Raises on digits that are no date."""
    day = f"printf('%02d', CAST({prefix}_DAY AS INTEGER))"
    return f"CAST(strptime({prefix}_YEARMONTH || {day}, '%Y%m%d') AS DATE)"


def dollars_sql(column: str) -> str:
    """SQL for the dollars in a Storm Data amount ('10.00K', '2.5M', '500'; a bare 'K' is 0), NULL when
    nothing was entered or it has another form."""
    amount = f'upper(trim({column}))'
    scale = (
        f"CASE right({amount}, 1) WHEN 'H' THEN 1e2 WHEN 'K' THEN 1e3 WHEN 'M' THEN 1e6 "
        "WHEN 'B' THEN 1e9 ELSE 1 END"
    )
    return (
        rf"CASE WHEN {amount} <> '' AND regexp_full_match({amount}, '(?:[0-9]+\.?[0-9]*|\.[0-9]+)?[HKMB]?') "
        f"THEN coalesce(TRY_CAST(regexp_extract({amount}, '^[0-9.]+') AS DOUBLE), 0) * {scale} END"
    )


def title_sql(column: str) -> str:
    """SQL for `column` in title case: every run of letters capitalised, a possessive 's kept lower
    ("PRINCE GEORGE'S" -> "Prince George's", "O'BRIEN" -> "O'Brien")."""
    runs = rf"regexp_extract_all(lower({column}), '\p{{L}}+(?:''s\b)?|[^\p{{L}}]+')"
    return f"array_to_string(list_transform({runs}, lambda run: upper(run[1]) || run[2:]), '')"


def reports(
    con: duckdb.DuckDBPyConnection, csv_paths: Iterable[str | Path]
) -> duckdb.DuckDBPyRelation:
    """The CONUS flood reports in the NCEI details CSVs at `csv_paths`, one file a year."""
    lon, lat = 'TRY_CAST(BEGIN_LON AS DOUBLE)', 'TRY_CAST(BEGIN_LAT AS DOUBLE)'
    property_dollars, crop_dollars = dollars_sql('DAMAGE_PROPERTY'), dollars_sql('DAMAGE_CROPS')
    paths = ', '.join(f"'{path}'" for path in csv_paths)
    types = ', '.join(f"'{event_type}'" for event_type in EVENT_TYPES)
    # Every column as text: the casts below name what the build reads. Not strict, because NCEI leaves quotes
    # inside quoted text unescaped ("S OF "7 MILE" BRIDGE").
    return con.sql(
        f"""
        SELECT *, {bbox_sql('geometry')} AS bbox FROM (
            SELECT CAST(EVENT_ID AS BIGINT) AS event_id,
                   CAST(EPISODE_ID AS BIGINT) AS episode_id,
                   EVENT_TYPE AS type,
                   CAST({date_sql('BEGIN')} AS VARCHAR) AS begin_date,
                   CAST({date_sql('END')} AS VARCHAR) AS end_date,
                   year({date_sql('BEGIN')}) AS year,
                   {title_sql('STATE')} AS state,
                   {title_sql('CZ_NAME')} AS county,
                   {title_sql('BEGIN_LOCATION')} AS location,
                   FLOOD_CAUSE AS cause,
                   CAST(round(coalesce({property_dollars} + {crop_dollars}, {property_dollars},
                                       {crop_dollars})) AS BIGINT) AS damage,
                   CAST(DEATHS_DIRECT AS INTEGER) + CAST(DEATHS_INDIRECT AS INTEGER) AS deaths,
                   CAST(INJURIES_DIRECT AS INTEGER) + CAST(INJURIES_INDIRECT AS INTEGER) AS injuries,
                   CAST(END_LON AS DOUBLE) AS end_lon,
                   CAST(END_LAT AS DOUBLE) AS end_lat,
                   ST_Point({lon}, {lat}) AS geometry
            FROM read_csv([{paths}], header=true, all_varchar=true, quote='"', escape='"', strict_mode=false)
            WHERE EVENT_TYPE IN ({types})
              AND {lon} BETWEEN {CONUS.left} AND {CONUS.right} AND {lat} BETWEEN {CONUS.bottom} AND {CONUS.top}
        )
        """
    )


def write_local(
    csv_paths: Iterable[str | Path],
    out_dir: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
    temp_dir: str | Path | None = None,
) -> Path:
    """Write the CONUS flood reports from the NCEI details CSVs as GeoParquet with overviews to
    out_dir/storm_events_floods.parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'storm_events_floods.parquet'
    meta = processing_metadata() | dict(metadata or {})
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp, duckdb_connect() as con:
        # geoparquet-lod reads its source several times, so it needs a file, not a stream.
        staged = Path(tmp) / 'floods.parquet'
        write_relation(reports(con, csv_paths), meta, staged)
        repeated = con.execute(
            'SELECT count(*) - count(DISTINCT event_id) FROM read_parquet($p)', {'p': str(staged)}
        ).fetchone()[0]
        if repeated:
            raise ValueError(f'{repeated} reports repeat an event_id; read each year from one file')
        geoparquet_lod.write(
            str(staged),
            out,
            zooms=ZOOMS,
            thinning=THINNING,
            carry=CARRY,
            metadata=meta,
            temp_dir=tmp,
        )
    return out


def save(chunks: Iterable[bytes], dest: Path) -> str:
    """Write `chunks` to `dest`; returns their sha256."""
    digest = hashlib.sha256()
    with dest.open('wb') as f:
        for chunk in chunks:
            digest.update(chunk)
            f.write(chunk)
    return digest.hexdigest()


def fetch_index() -> str:
    with urllib.request.urlopen(INDEX, timeout=60) as resp:
        return resp.read().decode()


def download(name: str, dest: Path) -> str:
    """Stream the details file `name` from NCEI to `dest`; returns its sha256."""
    with urllib.request.urlopen(INDEX + name, timeout=300) as resp:
        return save(iter(lambda: resp.read(1 << 20), b''), dest)


def read_mirror(name: str, dest: Path) -> str:
    """Stream the mirrored details file `name` to `dest`; returns its sha256."""
    return save(obstore.get(StorageConfig().object_store(), raw_key(name)).stream(), dest)


def exists(key: str) -> bool:
    try:
        obstore.head(StorageConfig().object_store(), key)
    except FileNotFoundError:
        return False
    return True


def main(overwrite: bool = typer.Option(False, '--overwrite')) -> None:
    names = list(latest_files(fetch_index()).values())
    if not names:
        raise ValueError(f'{INDEX} lists no details files')
    key = key_for(source_date(names))
    if exists(key) and not overwrite:
        typer.echo(f'skip: {key} exists; pass --overwrite')
        return
    # TMPDIR, else the system temp dir; on Coiled pass --env TMPDIR=/scratch (VM disk, not overlay /tmp).
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / 'raw'
        raw.mkdir()
        mirrored = {name for name in names if exists(raw_key(name))}
        sha256 = {
            name: (read_mirror if name in mirrored else download)(name, raw / name)
            for name in names
        }
        meta = source_metadata(sha256, retrieved=datetime.now(UTC).date())
        out = write_local(
            [raw / name for name in names], Path(tmp) / 'out', metadata=meta, temp_dir=tmp
        )
        # After the build, so a failed one leaves no mirror behind.
        for name in names:
            if name not in mirrored:
                obstore.put(StorageConfig().object_store(), raw_key(name), raw / name)
                typer.echo(f'mirrored {raw_key(name)}')
        obstore.put(StorageConfig().object_store(), key, out)
    typer.echo(f'wrote {key}')


if __name__ == '__main__':
    typer.run(main)
