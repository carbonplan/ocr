import json
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import odc.geo.xr  # noqa: F401
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from odc.geo import CRS
from odc.geo.geom import BoundingBox

from ocr.config import OVERTURE_RELEASE, TIGER_VINTAGE
from ocr.duck import duckdb_connect
from ocr.geoparquet import write_relation
from ocr.regions import RegionLevel

CONUS = BoundingBox(-125.0, 24.0, -66.0, 50.0, crs='EPSG:4326')

OVERTURE_LICENSE = 'ODbL-1.0'  # Buildings dataset; includes OpenStreetMap data
OVERTURE_ATTRIBUTION_URL = 'https://docs.overturemaps.org/attribution/'
OVERTURE_ATTRIBUTION = 'OpenStreetMap contributors, Overture Maps Foundation'
# Fire's OCRConfig.vector.metadata_dict, minus `version` (comes with the versioned storage layout).
VECTOR_METADATA = {
    'provider': 'CarbonPlan',
    'terms_of_access': 'https://docs.carbonplan.org/ocr/en/latest/terms-of-data-access.html',
    'data_sources': 'https://docs.carbonplan.org/ocr/en/latest/reference/data-sources.html',
    'license_name': 'ODBL',
    'license_url': 'https://opendatacommons.org/licenses/odbl/',
    'notice': 'Contains information from the Overture Maps Foundation database, which is made available here under the Open Database License (ODbL), a copy of which is available at https://opendatacommons.org/licenses/odbl/1-0/.',
}


class PointSamplingStrategy(StrEnum):
    """How to pick one point per building footprint (the DuckDB function name).
    Either CENTROID, which can sample outside of a building's extent for strange shaped polygons,
    or ON_SURFACE, which guarantees in-polygon sampling, but not centroid."""

    CENTROID = 'ST_Centroid'
    ON_SURFACE = 'ST_PointOnSurface'


class Axis(NamedTuple):
    """Start and spacing of an evenly spaced coordinate."""

    start: float
    step: float


def _axis(coord: np.ndarray) -> Axis:
    """Axis of an evenly spaced coordinate. Errors if spacing is uneven."""
    step = float(coord[-1] - coord[0]) / (len(coord) - 1)
    expected = coord[0] + step * np.arange(len(coord))
    if np.abs(coord - expected).max() > 0.1 * abs(step):
        raise ValueError('coordinate is not evenly spaced')
    return Axis(float(coord[0]), step)


def _validate_dataset(ds: xr.Dataset) -> tuple[CRS, Axis, Axis]:
    """Check `ds` is a 2D grid with a CRS and evenly spaced coords; return its CRS and (y, x) axes."""
    if (crs := ds.odc.crs) is None:
        raise ValueError('ds has no CRS; write CF attrs (xproj write_crs_info)')
    ydim, xdim = ds.odc.spatial_dims
    if not_2d := [v for v, da in ds.data_vars.items() if set(da.dims) != {ydim, xdim}]:
        raise ValueError(f'variables {not_2d} are not 2D over {(ydim, xdim)}')
    # Start and spacing of each axis, so SQL can turn a point into row/column.
    return crs, _axis(ds[ydim].values), _axis(ds[xdim].values)


def _cell_index_sql(coord: str, axis: Axis) -> str:
    """Same cell as .sel(method='nearest') on an evenly spaced grid (coords are cell centres)."""
    return f'round(({coord} - {axis.start}) / {axis.step})::BIGINT'


def _sample(ds: xr.Dataset, iy: np.ndarray, ix: np.ndarray) -> dict[str, pa.Array]:
    """Sample a raster at every Overture building in CONUS.

    The ds.sel(lat=..., lon=..., method='nearest') approach was slow at
    CONUS scale b/c for each point it searches the coordinate array for the closest
    value. B/c our grids are evenly spaced, the cell can be computed directly instead.
    DuckDB does that arithmetic while it reads the buildings, and Xarray then fetches
    every building's value in one pointwise ds.isel() call."""

    ydim, xdim = ds.odc.spatial_dims
    inside = (iy >= 0) & (iy < ds.sizes[ydim]) & (ix >= 0) & (ix < ds.sizes[xdim])
    pts = ds.isel(
        {
            ydim: xr.DataArray(np.where(inside, iy, 0), dims='pt'),
            xdim: xr.DataArray(np.where(inside, ix, 0), dims='pt'),
        }
    )
    # NaN -> NULL: pyarrow's stats skip NaN, so DuckDB NaN filters (region stats, lod_shown) miss it (duckdb#25521).
    return {
        str(v): pa.array(da.values, mask=~inside | np.isnan(da.values))
        for v, da in pts.data_vars.items()
    }


def sample_buildings(
    ds: xr.Dataset,
    out: str | Path,
    *,
    blocks: str | Path | None = None,
    source: str | Path | None = None,
    release: str = OVERTURE_RELEASE,
    point_strategy: PointSamplingStrategy = PointSamplingStrategy.CENTROID,
    bbox: tuple[float, float, float, float] = CONUS.bbox,
    temp_dir: str | Path | None = None,
) -> None:
    """Sample a 2D xarray dataset at every Overture building in a CONUS block and write GeoParquet to `out`
    (local path or s3://). Point sampling, not zonal stats.

    `ds` must be a 2D grid with a CRS and evenly spaced coords; sampling points move into its CRS. Open it
    with chunks=None; dask is about 3.5x slower! Pick vars first with ds[[...]].

    Each building gets GEOID, the TIGER block (`blocks`, default RegionLevel.BLOCK.uri) holding its
    sampling point; buildings in no block (Canada, Mexico, offshore) are dropped. A point on a shared
    block edge keeps the lowest GEOID. Buildings off the raster get null; raster NaN stays NaN.
    `source` overrides the Overture buildings URL. `bbox` (lon/lat xmin, ymin, xmax, ymax; default CONUS)
    keeps buildings whose bbox lies inside it, e.g. a city for a dev run.

    Columns: id (GERS ID), GEOID, one per var, geometry and bbox. Metadata: VECTOR_METADATA, the Overture
    release, its license, attribution (+ url), TIGER vintage and `ds.attrs` (as dataset_<name>).

    Streams through local files in `temp_dir`, so peak RAM is the cell indexes and sampled values, not
    the table: DuckDB writes the join, `_sample` reads only iy/ix, then a positional join puts them together.

    >>> ds = xr.open_zarr(icechunk_store, chunks=None, consolidated=False)
    >>> sample_buildings(ds[['rps_2011']], 'buildings.parquet')
    """

    point_strategy = PointSamplingStrategy(point_strategy)
    ds_crs, y, x = _validate_dataset(ds)
    left, bottom, right, top = bbox
    if not (left < right and bottom < top):
        raise ValueError(f'bbox {bbox} is not (xmin, ymin, xmax, ymax)')

    url = str(
        source
        or f's3://overturemaps-us-west-2/release/{release}/theme=buildings/type=building/*.parquet'
    )
    params = {
        'url': url,
        'blocks': str(blocks or RegionLevel.BLOCK.uri),
        'left': left,
        'bottom': bottom,
        'right': right,
        'top': top,
    }
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp, duckdb_connect() as con:
        joined, values = Path(tmp) / 'joined.parquet', Path(tmp) / 'values.parquet'
        # Relabels blocks to the buildings' CRS. Same as ST_Transform here (PROJ's default NAD83 -> WGS84 is a
        # null transform), so both ignore the real ~1-1.5 m datum offset. ponytail: a projected block CRS needs ST_Transform.
        crs = con.sql(
            'SELECT ST_CRS(geometry) FROM read_parquet($url) LIMIT 1', params={'url': url}
        ).fetchall()[0][0]
        geom = f"ST_SetCRS(geometry, '{crs}')" if crs else 'geometry'
        # Sampling point in the raster's CRS (lon/lat in, x/y out). Blocks still join on the lon/lat point.
        p = 'c' if ds_crs == CONUS.crs else f"ST_Transform(c, 'OGC:CRS84', '{ds_crs}', true)"
        con.execute(
            f"""
            COPY (
                WITH blk AS (SELECT GEOID, {geom} AS geometry FROM read_parquet($blocks))
                SELECT b.id, blk.GEOID, b.geometry, b.bbox,
                       {_cell_index_sql('ST_Y(b.p)', y)} AS iy,
                       {_cell_index_sql('ST_X(b.p)', x)} AS ix
                FROM (SELECT id, geometry, bbox, {point_strategy}(geometry) AS c, {p} AS p
                      FROM read_parquet($url)
                      WHERE bbox.xmin >= $left AND bbox.ymin >= $bottom
                        AND bbox.xmax <= $right AND bbox.ymax <= $top) b
                JOIN blk ON ST_Intersects(blk.geometry, b.c)
            ) TO $joined (FORMAT parquet)
            """,
            params | {'joined': str(joined)},
        )
        # Read the raster value at every row's cell in one batch, in the joined file's row order.
        idx = pq.read_table(joined, columns=['iy', 'ix'])
        pq.write_table(pa.table(_sample(ds, idx['iy'].to_numpy(), idx['ix'].to_numpy())), values)
        del idx
        # A point on a shared block edge joins twice (1 in 20.4M in Texas); keep the row with
        # the lowest GEOID. By row number, so one row per id even if GEOIDs tie.
        rel = con.sql(
            """
            WITH j AS (SELECT * FROM read_parquet($joined, file_row_number = true)),
            dups AS (
                SELECT id, arg_min(file_row_number, GEOID) AS keep
                FROM j GROUP BY id HAVING count(*) > 1
            )
            SELECT j.id, j.GEOID, v.*, j.geometry, j.bbox
            FROM j POSITIONAL JOIN read_parquet($values) v
            LEFT JOIN dups USING (id)
            WHERE dups.id IS NULL OR j.file_row_number = dups.keep
            """,
            params={'joined': str(joined), 'values': str(values)},
        )
        write_relation(rel, _metadata(ds, release, point_strategy), out)


def _metadata(
    ds: xr.Dataset, release: str, point_strategy: PointSamplingStrategy
) -> dict[str, str]:
    """Output key/value metadata. `ds.attrs` values must be text, so non-strings become JSON."""
    return (
        VECTOR_METADATA
        | {
            'overture_release': release,
            'overture_license': OVERTURE_LICENSE,
            'overture_attribution': OVERTURE_ATTRIBUTION,
            'overture_attribution_url': OVERTURE_ATTRIBUTION_URL,
            'point_strategy': str(point_strategy),
            'tiger_vintage': TIGER_VINTAGE,
        }
        | {f'dataset_{k}': v if isinstance(v, str) else json.dumps(v) for k, v in ds.attrs.items()}
    )
