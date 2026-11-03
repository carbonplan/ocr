"""Region GeoParquet laid out for the web map's range-request lookups (ported from carbonplan/ocr-multi-hazard#8).

    <out_key>/nation.parquet, states.parquet
    <out_key>/counties/state_fips=SS/data.parquet
    <out_key>/tracts/state_fips=SS/county_fips=CCC/data.parquet
    <out_key>/blocks/state_fips=SS/county_fips=CCC/data.parquet

Rows are in GEOID order with a row cap per page, so every column breaks pages at the same rows,
and GEOID and bbox carry statistics and page indexes. A reader finds a region by GEOID from the
page statistics, or by point from the bbox ones, and reads one page. Partitions keep each file's
footer and page indexes small, so opening a file costs one small read.
"""

import json
import tempfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import obstore
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from ocr.config import StorageConfig
from ocr.regions import RegionLevel

PAGE_ROWS = {'nation': 1, 'states': 1, 'counties': 1, 'tracts': 8, 'blocks': 32}
# GEOID prefix each level's files split on: counties by state, tracts and blocks by county.
SPLIT = {'counties': 2, 'tracts': 5, 'blocks': 5}
# Columns a lookup shows come first, so its reads sit close together.
FIRST = ['GEOID', 'STUSPS', 'NAME', 'building_count']
BBOX = ('xmin', 'ymin', 'xmax', 'ymax')


def _prepare(t: pa.Table, shown: Sequence[str]) -> pa.Table:
    """`t` with FIRST, then `shown`, then the rest, then bbox and geometry."""
    first = [c for c in dict.fromkeys([*FIRST, *shown]) if c in t.column_names]
    rest = [c for c in t.column_names if c not in first and c not in ('bbox', 'geometry')]
    return t.select([*first, *rest, 'bbox', 'geometry'])


def _geo(source_geo: dict, t: pa.Table) -> dict:
    """GeoParquet 1.1 metadata for `t`: the source's geometry column plus the bbox covering."""
    column = dict(source_geo['columns']['geometry'])
    column['covering'] = {'bbox': {k: ['bbox', k] for k in BBOX}}
    column['bbox'] = [
        pc.min(pc.struct_field(t['bbox'], 'xmin')).as_py(),
        pc.min(pc.struct_field(t['bbox'], 'ymin')).as_py(),
        pc.max(pc.struct_field(t['bbox'], 'xmax')).as_py(),
        pc.max(pc.struct_field(t['bbox'], 'ymax')).as_py(),
    ]
    return {'version': '1.1.0', 'primary_column': 'geometry', 'columns': {'geometry': column}}


def _write(
    t: pa.Table, path: Path, rows_per_page: int, source_geo: dict, meta: dict[str, str]
) -> None:
    """`t` as one row group, `rows_per_page` rows per page."""
    schema = t.schema.with_metadata(meta | {'geo': json.dumps(_geo(source_geo, t))})
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(
        path,
        schema,
        compression='zstd',
        compression_level=9,
        use_dictionary=False,
        write_page_index=True,
        write_statistics=['GEOID', *(f'bbox.{k}' for k in BBOX)],
        max_rows_per_page=rows_per_page,
        write_batch_size=min(rows_per_page, 1024),
        data_page_size=1 << 20,
    ) as writer:
        writer.write_table(t.cast(schema), row_group_size=max(t.num_rows, 1))


def _split(t: pa.Table, width: int) -> list[pa.Table]:
    """`t` (in GEOID order) split where the first `width` GEOID characters change."""
    key = pc.utf8_slice_codeunits(t['GEOID'], 0, width).to_numpy(zero_copy_only=False)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    return [t.slice(s, e - s) for s, e in zip(starts, ends, strict=True)]


def _parts(src: Path, level: RegionLevel) -> Iterator[tuple[str, pa.Table]]:
    """(relative path, rows) per output file, one state read at a time so blocks never sit in memory whole."""
    width = SPLIT.get(level)
    if width is None:
        yield f'{level}.parquet', pq.read_table(src).sort_by('GEOID')
        return
    ds = pads.dataset(src)
    states = sorted(
        set(pc.utf8_slice_codeunits(ds.to_table(columns=['GEOID'])['GEOID'], 0, 2).to_pylist())
    )
    for s in states:
        # src is GEOID-sorted (write_region_stats), so the range skips other states' row groups.
        state = ds.to_table(
            filter=(pads.field('GEOID') >= s) & (pads.field('GEOID') < s + '\uffff')
        )
        for part in _split(state.sort_by('GEOID'), width):
            g = part['GEOID'][0].as_py()
            hive = f'state_fips={g[:2]}' + (f'/county_fips={g[2:5]}' if width == 5 else '')
            yield f'{level}/{hive}/data.parquet', part


def write_level(
    src: str | Path,
    level: RegionLevel,
    out_key: str,
    shown: Sequence[str] = (),
    *,
    temp_dir: str | Path | None = None,
) -> None:
    """Rewrite `src` (a `write_region_stats` GeoParquet for `level`) in the web layout under storage key `out_key`.

    `shown` names the columns a lookup reads, placed right after the identifying ones. The source's
    key/value metadata (bins, tiger_vintage, Overture) is kept. `temp_dir` holds files until uploaded.
    """
    raw = pq.read_schema(src).metadata
    source_geo = json.loads(raw[b'geo'])
    meta = {k.decode(): v.decode() for k, v in raw.items() if k not in (b'geo', b'ARROW:schema')}
    store = StorageConfig().object_store()

    def put(rel: str, path: Path) -> None:
        obstore.put(store, f'{out_key}/{rel}', path)
        path.unlink()

    # Uploads run alongside the writes: tracts and blocks are ~3k small files each.
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp, ThreadPoolExecutor(16) as pool:
        uploads = []
        for rel, part in _parts(Path(src), level):
            path = Path(tmp) / rel
            _write(_prepare(part, shown), path, PAGE_ROWS[level], source_geo, meta)
            uploads.append(pool.submit(put, rel, path))
        for u in uploads:
            u.result()
