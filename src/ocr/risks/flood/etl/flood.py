"""Input data ETL for USGS Flood Damage Probability (FDP).

Converts the CONUS 100 m FDP GeoTIFF into group "0" of the flood output store (`OutputPaths.store`),
GeoZarr-tagged on its native EPSG:5070 grid (no reprojection; joins should move points into 5070,
not the raster out). Flood has no score step: FDP is the risk layer, so the ETL writes the store.

Raw tif provenance (manual, one-time):
1. Download Output_CONUS_FDP_100m.zip from SOURCE (ScienceBase only serves expiring presigned links).
   SOURCE: https://www.sciencebase.gov/catalog/item/6170694ed34ea36449a67ef7
2. Unzip Output_CONUS_FDP_100m/CONUS_FDP_100m.tif (striped GeoTIFF, 100 m, EPSG:5070).
3. Upload it to RAW_KEY.

Run: pixi run -e flood flood-etl --version v2.0.0 [--overwrite] (same as `ocr stage --version v2.0.0 flood etl`).
"""

import icechunk
import rioxarray
import xarray as xr
import xproj  # noqa: F401
from icechunk.xarray import to_icechunk
from topozarr import attach_geozarr_metadata, recommend_encoding

from ocr.config import OutputPaths, StorageConfig

# TODO: move RAW_KEY into a shared input-path config once a second risk has raw inputs.
SOURCE = 'https://www.sciencebase.gov/catalog/item/6170694ed34ea36449a67ef7'
RAW_KEY = 'input/flood/raw/usgs-fdp/CONUS_FDP_100m.tif'


def read_tif(cfg: StorageConfig) -> xr.Dataset:
    """Read the FDP tif `fdp` variable in EPSG:5070"""
    da = rioxarray.open_rasterio(cfg.uri(RAW_KEY), masked=True, chunks=None)
    da = da.squeeze('band', drop=True).astype('float32').drop_vars('spatial_ref')

    # Removes "GDAL STATISTICS_*"" which serialize to invalid JSON in zarr.json.
    da.attrs = {k: v for k, v in da.attrs.items() if not k.startswith('STATISTICS_')}
    ds = da.to_dataset(name='fdp')
    ds.attrs = {
        'title': 'USGS Flood Damage Probability (FDP), CONUS 100 m',
        'source': SOURCE,
        'doi': 'https://doi.org/10.5066/P954TTQN',
    }
    # write_crs_info stores CF crs_wkt on spatial_ref, so the CRS survives a reopen (odc reads it).
    return ds.proj.assign_crs(spatial_ref='EPSG:5070').proj.write_crs_info()


def write_icechunk(cfg: StorageConfig, ds: xr.Dataset, prefix: str, overwrite: bool) -> str:
    """Write ds to group "0" of the repo at `prefix` and return the snapshot id. Refuses an existing
    repo unless overwrite, which adds a commit on main."""
    storage = cfg.icechunk_storage(prefix)
    if icechunk.Repository.exists(storage) and not overwrite:
        raise FileExistsError(f'{cfg.uri(prefix)} exists; pass --overwrite')
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session('main')
    to_icechunk(ds, session, group='0', mode='w', encoding=recommend_encoding(ds))
    return session.commit('USGS FDP 100m EPSG:5070')


def run(out: OutputPaths, overwrite: bool = False) -> None:
    cfg = StorageConfig()
    ds = read_tif(cfg)
    # Adds GeoZarr metadata
    ds = attach_geozarr_metadata(ds)
    snapshot = write_icechunk(cfg, ds, out.store, overwrite)
    print(f'wrote {cfg.uri(out.store)} : {snapshot}')
