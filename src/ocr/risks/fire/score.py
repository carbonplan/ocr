"""Fire score: wind-adjusted risk per region, written into group "0" of `OutputPaths.store`.

One part per region of the Scott 2024 30 m chunk grid (`ChunkingConfig`). `parts` plans (creates
the template, returns regions not in the commit history); `write_region` writes and commits one
region. Writers don't coordinate: each commit rebases onto the others (regions never share chunks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import icechunk
import numpy as np
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor
from icechunk.xarray import to_icechunk
from topozarr import attach_geozarr_metadata, recommend_encoding

from ocr.config import OutputPaths, StorageConfig

if TYPE_CHECKING:
    from ocr.risks.fire.grid import ChunkingConfig

TEMPLATE = 'initialize store with template'


def _grid() -> ChunkingConfig:
    # Lazy: fire config opens the Scott 2024 grid from the catalog; keeps `ocr.stages` imports light.
    from ocr.risks.fire.grid import ChunkingConfig

    return ChunkingConfig()


def _write_template(session: icechunk.Session, grid: ChunkingConfig, out: OutputPaths) -> None:
    """Empty CONUS arrays (metadata only), one shard per region, with the CRS and variable attrs."""
    import dask.array

    from ocr.risks.fire.model import VARIABLE_ATTRS

    shape = (grid.ds.sizes['latitude'], grid.ds.sizes['longitude'])
    # Region-sized dask blocks, so recommend_encoding picks chunks that divide a region.
    chunks = (grid.chunks['latitude'], grid.chunks['longitude'])
    empty = dask.array.empty(shape, dtype='float32', chunks=chunks)
    ds = xr.Dataset(grid.ds.coords).drop_vars('spatial_ref')
    for name, attrs in VARIABLE_ATTRS.items():
        # Region writes drop attrs, so the template holds them.
        ds[name] = xr.DataArray(empty, dims=('latitude', 'longitude'), attrs=attrs)
    ds.attrs = {
        'version': f'v{out.version}',
        'provider': 'CarbonPlan',
        'terms_of_access': 'https://docs.carbonplan.org/ocr/en/latest/terms-of-data-access.html',
        'data_sources': 'https://docs.carbonplan.org/ocr/en/latest/reference/data-sources.html',
        'license_name': 'CC-BY-4.0',
        'license_url': 'https://creativecommons.org/licenses/by/4.0/',
    }
    # CF crs_wkt on spatial_ref, so a later buildings stage can sample the store.
    ds = ds.proj.assign_crs(spatial_ref='EPSG:4326').proj.write_crs_info()
    # As flood's group "0": GeoZarr attrs and topozarr's chunks. Shards are whole regions, so
    # concurrent region commits never share an object (zarr raises if chunks don't divide them).
    ds = attach_geozarr_metadata(ds, x_dim='longitude', y_dim='latitude')
    encoding = recommend_encoding(ds, x_dim='longitude', y_dim='latitude', chunks_per_shard=None)
    for name in VARIABLE_ATTRS:
        encoding[name] |= {'shards': chunks, 'fill_value': np.nan}
    ds.to_zarr(
        session.store, group='0', mode='w', compute=False, encoding=encoding, consolidated=False
    )
    session.commit(TEMPLATE)


def parts(out: OutputPaths, region_ids: list[str] | None = None) -> list[str]:
    """Region ids (all valid ones, or `region_ids`) not yet in the store's commit history.
    Creates the store and its template first if missing. Unknown ids and an empty list raise.
    """
    if region_ids == []:
        raise ValueError('region_ids is empty; omit it to plan every region')
    grid = _grid()
    valid = grid.valid_region_ids
    if unknown := sorted(set(region_ids or ()) - set(valid)):
        raise ValueError(f'unknown fire region ids: {unknown}')
    repo = icechunk.Repository.open_or_create(StorageConfig().icechunk_storage(out.store))
    history = list(repo.ancestry(branch='main'))
    if not any(s.message == TEMPLATE for s in history):
        _write_template(repo.writable_session('main'), grid, out)
    done = {s.metadata.get('region_id') for s in history}
    return [r for r in (region_ids or valid) if r not in done]


def write_region(out: OutputPaths, region_id: str) -> None:
    """Compute one region and commit it into group "0" of the store `parts` created."""
    from ocr.risks.fire.model import calculate_wind_adjusted_risk

    grid = _grid()
    y_slice, x_slice = grid.region_id_to_latlon_slices(region_id=region_id)
    ds = calculate_wind_adjusted_risk(y_slice=y_slice, x_slice=x_slice).chunk(grid.chunks)
    repo = icechunk.Repository.open(StorageConfig().icechunk_storage(out.store))
    session = repo.writable_session('main')
    to_icechunk(ds, session, group='0', region='auto')
    # region_id in commit metadata, not parsed from the message; icechunk 2.x keeps both in one
    # repo info file, so resume is one read. Contended commits rebase (up to 1000 tries); a real
    # error fails the task, and the next round re-plans it.
    session.commit(
        f'wrote region_id ({region_id})',
        metadata={'region_id': region_id},
        rebase_with=icechunk.ConflictDetector(),
    )
