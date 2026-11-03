"""Shared GeoParquet read/write helpers (DuckDB spatial)."""

from pathlib import Path

import duckdb
import pyarrow as pa
from upath import UPath

from ocr.duck import duckdb_connect


def write_geoparquet(tbl: pa.Table | pa.RecordBatchReader, out: str | Path) -> None:
    """Save `tbl` as GeoParquet (local path or s3://), keeping its metadata. A reader streams."""
    meta = {k.decode(): v.decode() for k, v in (tbl.schema.metadata or {}).items()}
    upath = UPath(out)
    local = upath.protocol in ('', 'file')
    # Local writes go to .tmp then rename, so a killed run never leaves a partial file at `out`.
    # S3 needs no tmp: a multipart upload only appears once complete.
    dest = Path(f'{upath.path}.tmp') if local else out
    if local:
        dest.parent.mkdir(parents=True, exist_ok=True)
    with duckdb_connect(s3_write=upath.protocol == 's3') as con:
        con.register('tbl', tbl)
        # KV_METADATA needs a non-empty STRUCT; {} binds as a MAP and errors.
        kv = ', KV_METADATA $meta' if meta else ''
        con.execute(
            f"COPY tbl TO $out (FORMAT parquet, GEOPARQUET_VERSION 'V2'{kv})",
            {'out': str(dest)} | ({'meta': meta} if meta else {}),
        )
    if local:
        dest.replace(upath.path)


def write_relation(rel: duckdb.DuckDBPyRelation, meta: dict[str, str], out: str | Path) -> None:
    """Stream `rel` to GeoParquet with `meta`. Call inside its connection: the reader dies when it closes."""
    reader = rel.to_arrow_reader()
    write_geoparquet(
        pa.RecordBatchReader.from_batches(reader.schema.with_metadata(meta), reader), out
    )


def kv_metadata(con: duckdb.DuckDBPyConnection, path: str) -> dict[str, str]:
    """The parquet's own key/value metadata, minus what the GeoParquet writer regenerates."""
    return dict(
        con.execute(
            "SELECT decode(key), decode(value) FROM parquet_kv_metadata($p) WHERE decode(key) NOT IN ('geo', 'ARROW:schema')",
            {'p': path},
        ).fetchall()
    )


def bbox_sql(geom: str) -> str:
    """SQL for a GeoParquet bbox struct of `geom`: {xmin, ymin, xmax, ymax}."""
    return (
        f"{{'xmin': ST_XMin({geom}), 'ymin': ST_YMin({geom}), "
        f"'xmax': ST_XMax({geom}), 'ymax': ST_YMax({geom})}}"
    )
