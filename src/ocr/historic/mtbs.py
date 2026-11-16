"""MTBS burned-area perimeters (USGS/USFS Monitoring Trends in Burn Severity), CONUS, as GeoParquet with overviews.

writes input/historic/mtbs/<YYYYMMDD>/mtbs_perims.parquet, dated by the source's Last-Modified so a
rebuild after an upstream update never overwrites an earlier file. the downloaded zip is mirrored beside it
at raw/<sha256[:16]>/mtbs_perimeter_data.zip (content-addressed and never overwritten, uploaded after the
build succeeds), and the output's metadata records that key. every incident type is kept (no
filter on incid_type). geometry is made valid, reduced to polygons, moved from NAD83 (EPSG:4269) to
WGS84 and simplified at ~0.1 m, which strips the 1 m densified vertices

run: pixi run historic-mtbs [--overwrite]
peak is ~29 GiB and ~5 min on 16 vCPU / 124 GiB:
coiled run --vm-type r8g.4xlarge --region us-west-2 --no-credential-forwarding --env TMPDIR=/scratch pixi run historic-mtbs
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
import geoparquet_lod
import obstore
import typer

from ocr.buildings import CONUS
from ocr.config import StorageConfig
from ocr.duck import duckdb_connect
from ocr.geoparquet import bbox_sql, kv_metadata, write_relation

URL = (
    'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/'
    'burned_area_extent_shapefile/mtbs_perimeter_data.zip'
)
SHAPEFILE = 'mtbs_perims_DD.shp'
ATTRIBUTION = 'MTBS (USGS/USFS)'
LICENSE = 'Public domain (US Government work)'
# Past z11 the writer makes millions of copies of polygons that straddle tile edges.
ZOOMS = list(range(12))
# Zoom from which every row is shown, whatever the thinning says. 4 is what 'auto' picked on the real data, but
# only an explicit int forces it. geoparquet-lod doesn't record it, so processing_metadata does.
COMPLETE_FROM = 4
# tippecanoe's --drop-smallest-as-needed, which the old PMTiles pipeline used. Swap it here.
THINNING = geoparquet_lod.drop_smallest()
# Fields copied into the overviews for styling; the web reads the rest from the rows.
CARRY = ['year']
# pyproject.toml [tool.setuptools_scm] fallback_version: what a build without repository metadata reports.
SCM_FALLBACK_VERSION = '999'
# 1e-6 degrees is ~0.1 m.
SIMPLIFY_TOLERANCE = 1e-6


def key_for(day: date) -> str:
    return f'input/historic/mtbs/{day:%Y%m%d}/mtbs_perims.parquet'


def source_date(last_modified: str | None, *, today: date | None = None) -> date:
    """The HTTP Last-Modified header's date (in UTC), or `today` when the header is missing."""
    if not last_modified:
        return today or datetime.now(UTC).date()
    return parsedate_to_datetime(last_modified).astimezone(UTC).date()


def raw_key(day: date, sha256: str) -> str:
    """Content-addressed, so two different zips published on one date get two mirrors, never one replaced."""
    return f'input/historic/mtbs/{day:%Y%m%d}/raw/{sha256[:16]}/mtbs_perimeter_data.zip'


def source_metadata(
    url: str, *, last_modified: str | None, sha256: str, retrieved: date, raw_mirror: str
) -> dict[str, str]:
    return {
        'mtbs_url': url,
        'mtbs_last_modified': last_modified or '',
        'mtbs_sha256': sha256,
        'mtbs_raw_mirror': raw_mirror,
        'mtbs_retrieved': retrieved.isoformat(),
        'mtbs_attribution': ATTRIBUTION,
        'mtbs_license': LICENSE,
    }


def ocr_version() -> str:
    """The ocr version, whose commit also pins geoparquet-lod in pyproject.toml."""
    version = importlib_metadata.version('ocr')
    return f'unknown ({version})' if version == SCM_FALLBACK_VERSION else version


def processing_metadata() -> dict[str, str]:
    """What shaped the output that geoparquet-lod doesn't already record (it writes zooms, carried fields and thinning)."""
    return {
        'mtbs_ocr_version': ocr_version(),
        'mtbs_simplify_tolerance': str(SIMPLIFY_TOLERANCE),
        'mtbs_complete_from': str(COMPLETE_FROM),
    }


def shapefile_source(zip_path: str | Path) -> str:
    """SQL table expression reading the perimeter shapefile inside the zip without unzipping (geometry column `geom`)."""
    return f"ST_Read('/vsizip/{zip_path}/{SHAPEFILE}')"


def perimeters(con: duckdb.DuckDBPyConnection, source: str) -> duckdb.DuckDBPyRelation:
    """The cleaned CONUS perimeters from `source`, a table or table expression with the MTBS columns and `geom`
    in EPSG:4269 (see `shapefile_source`)."""
    envelope = f'ST_MakeEnvelope({CONUS.left}, {CONUS.bottom}, {CONUS.right}, {CONUS.top})'
    # Order matters. MakeValid returns GEOMETRYCOLLECTIONs (polygons plus stray lines/points) for some rows and
    # collapses exact-duplicate parts; CollectionExtract keeps the polygons. always_xy=true, else the
    # transform swaps lat/lon. Simplify last, in degrees, after the transform.
    geometry = (
        'ST_SimplifyPreserveTopology('
        "ST_Transform(ST_CollectionExtract(ST_MakeValid(geom), 3), 'EPSG:4269', 'EPSG:4326', true), "
        f'{SIMPLIFY_TOLERANCE})'
    )
    return con.sql(
        f"""
        SELECT *, {bbox_sql('geometry')} AS bbox FROM (
            SELECT event_id, incid_name, incid_type,
                   ig_date,  -- the published string, as in the old PMTiles
                   year(CAST(ig_date AS DATE)) AS year,
                   burnbndac, asmnt_type,
                   {geometry} AS geometry
            FROM {source}
            WHERE ST_Intersects(geom, {envelope})
        )
        """
    )


def write_local(
    zip_path: str | Path,
    out_dir: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
    temp_dir: str | Path | None = None,
) -> Path:
    """Write the cleaned perimeters from the MTBS zip as GeoParquet with overviews to out_dir/mtbs_perims.parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'mtbs_perims.parquet'
    meta = processing_metadata() | dict(metadata or {})
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp, duckdb_connect() as con:
        # geoparquet-lod reads its source several times, so it needs a file, not a stream.
        staged = Path(tmp) / 'perims.parquet'
        write_relation(perimeters(con, shapefile_source(zip_path)), meta, staged)
        geoparquet_lod.write(
            str(staged),
            out,
            zooms=ZOOMS,
            thinning=THINNING,
            carry=CARRY,
            complete_from=COMPLETE_FROM,
            metadata=meta,
            temp_dir=tmp,
        )
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
        return kv_metadata(con, StorageConfig().uri(key)).get('mtbs_sha256')


def main(overwrite: bool = typer.Option(False, '--overwrite')) -> None:
    # TMPDIR, else the system temp dir; on Coiled pass --env TMPDIR=/scratch (VM disk, not overlay /tmp).
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / 'mtbs_perimeter_data.zip'
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
        out = write_local(zip_path, Path(tmp) / 'out', metadata=meta, temp_dir=tmp)
        # After the build, so a failed one leaves no mirror behind.
        mirror(zip_path, raw)
        obstore.put(StorageConfig().object_store(), key, out)
    typer.echo(f'wrote {key}')


if __name__ == '__main__':
    typer.run(main)
