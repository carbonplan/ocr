"""Look up hazard raster values for every Overture building in CONUS.

1. DuckDB reads the buildings, picks one point per footprint and works out
   which raster cell (row, column) that point falls in.
2. xarray reads the value at each of those cells in one batch.
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import odc.geo.xr  # noqa: F401
import pyarrow as pa
import xarray as xr
from upath import UPath

from ocr.duck import duckdb_connect
from ocr.extent import CONUS

OVERTURE_RELEASE = '2026-09-23.1'


class PointSamplingStrategy(StrEnum):
    """How to pick one point per building footprint (the DuckDB function name)."""

    CENTROID = 'ST_Centroid'
    ON_SURFACE = 'ST_PointOnSurface'

    @classmethod
    def _missing_(cls, value: object) -> None:
        options = ', '.join(f'{cls.__name__}.{m.name}' for m in cls)
        raise ValueError(f'{value!r} is not a valid {cls.__name__}; use one of {options}')


class Axis(NamedTuple):
    """Start and spacing of an evenly spaced coordinate."""

    start: float
    step: float


def _grid(coord: np.ndarray) -> Axis:
    """Axis of an evenly spaced coordinate. Errors if spacing is uneven."""
    step = float(coord[-1] - coord[0]) / (len(coord) - 1)
    expected = coord[0] + step * np.arange(len(coord))
    if np.abs(coord - expected).max() > 0.1 * abs(step):
        raise ValueError('coordinate is not evenly spaced')
    return Axis(float(coord[0]), step)


def _validate_dataset(ds: xr.Dataset) -> tuple[Axis, Axis]:
    """Check `ds` is a 2D EPSG:4326 grid with evenly spaced coords; return its (y, x) axes."""
    if ds.odc.crs != CONUS.crs:
        raise ValueError(f'ds CRS {ds.odc.crs} is not {CONUS.crs}')
    ydim, xdim = ds.odc.spatial_dims
    if not_2d := [v for v, da in ds.data_vars.items() if set(da.dims) != {ydim, xdim}]:
        raise ValueError(f'variables {not_2d} are not 2D over {(ydim, xdim)}')
    # Start and spacing of each axis, so SQL can turn lon/lat into row/column.
    return _grid(ds[ydim].values), _grid(ds[xdim].values)


def _cell(coord: str, axis: Axis) -> str:
    """SQL for the nearest cell index of `coord`: (coord - start) / step, rounded."""
    return f'round(({coord} - {axis.start}) / {axis.step})::BIGINT'


def _sample(ds: xr.Dataset, iy: np.ndarray, ix: np.ndarray) -> dict[str, pa.Array]:
    """Value of every var at each (row iy, column ix). Cells off the raster get null, keeping dtype."""
    ydim, xdim = ds.odc.spatial_dims
    inside = (iy >= 0) & (iy < ds.sizes[ydim]) & (ix >= 0) & (ix < ds.sizes[xdim])
    pts = ds.isel(
        {
            ydim: xr.DataArray(np.where(inside, iy, 0), dims='pt'),
            xdim: xr.DataArray(np.where(inside, ix, 0), dims='pt'),
        }
    )
    return {str(v): pa.array(da.values, mask=~inside) for v, da in pts.data_vars.items()}


def write_patable_to_gpq(tbl: pa.Table, out: str | Path) -> None:
    """Save `tbl` as GeoParquet (local path or s3://), keeping its metadata."""
    meta = {k.decode(): v.decode() for k, v in (tbl.schema.metadata or {}).items()}
    to_s3 = UPath(out).protocol == 's3'
    con = duckdb_connect(s3_write=to_s3)
    con.register('tbl', tbl)
    con.execute(
        "COPY tbl TO $out (FORMAT parquet, GEOPARQUET_VERSION 'V2', KV_METADATA $meta)",
        {'out': str(out), 'meta': meta},
    )


def sample_buildings(
    ds: xr.Dataset,
    *,
    release: str = OVERTURE_RELEASE,
    point_strategy: PointSamplingStrategy = PointSamplingStrategy.CENTROID,
) -> pa.Table:
    """Get the value of every var in `ds` for each Overture building inside CONUS.

    `ds` must be a 2D lat/lon (EPSG:4326) grid with evenly spaced coords. Open it
    with chunks=None; dask is about 3.5x slower. Pick vars first with ds[[...]].

    Everything is held in memory (CONUS: 179M rows, about 5 min on r9g.8xlarge).
    Buildings in the CONUS box but off the raster (Canada, Mexico) get null; raster NaN stays NaN.
    PointSamplingStrategy.CENTROID can land outside an L-shaped building;
    PointSamplingStrategy.ON_SURFACE can't.

    Returns columns id (GERS ID), one per var, geometry and bbox. The Overture
    release and `ds.attrs` (as dataset_<name>) are stored in the table metadata.

    >>> ds = xr.open_zarr(icechunk_store, chunks=None, consolidated=False)
    >>> tbl = sample_buildings(ds[['rps_2011']])
    >>> write_patable_to_gpq(tbl, 's3://bucket/buildings.parquet')
    """

    # Validation of dataset inputs. A plain str would go straight into the SQL.
    point_strategy = PointSamplingStrategy(point_strategy)
    y, x = _validate_dataset(ds)

    con = duckdb_connect()
    url = f's3://overturemaps-us-west-2/release/{release}/theme=buildings/type=building/*.parquet'
    # Numbers go straight into the SQL text: $param placeholders made DuckDB use one core.
    tbl = con.sql(
        f"""
        SELECT id, geometry, bbox,
               {_cell('ST_Y(c)', y)} AS iy,
               {_cell('ST_X(c)', x)} AS ix
        FROM (SELECT id, geometry, bbox, {point_strategy}(geometry) AS c
              FROM read_parquet('{url}')
              WHERE bbox.xmin >= {CONUS.left} AND bbox.ymin >= {CONUS.bottom}
                AND bbox.xmax <= {CONUS.right} AND bbox.ymax <= {CONUS.top})
        """
    ).to_arrow_table()
    # Read the raster value at every building's cell in one batch.
    sampled = _sample(ds, tbl['iy'].to_numpy(), tbl['ix'].to_numpy())
    # Add one value column per var (same row order), then drop row/column.
    for var, col in sampled.items():
        tbl = tbl.append_column(var, col)
    tbl = tbl.select(['id', *sampled, 'geometry', 'bbox'])
    # Parquet metadata only holds text, so non-text attrs are stored as JSON.
    meta = {'overture_release': release} | {
        f'dataset_{k}': v if isinstance(v, str) else json.dumps(v) for k, v in ds.attrs.items()
    }
    return tbl.replace_schema_metadata(meta)
