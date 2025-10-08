# Script to modify the 2011 and 2014 USFS climate run datasets:
# - Reproject from EPSG:5070 to EPSG:4326 (X/Y -> lat/lon)
# - Interpolate from 270m to 30m
# - Note: This uses the USFS Community Risk 30m dataset as an input for interpolation

import coiled
import dask.base
import icechunk
import rich
import xarray as xr
from icechunk.xarray import to_icechunk
from odc.geo.xr import assign_crs, xr_reproject

from ocr import catalog

console = rich.console.Console()


def load_climate_run_ds(climate_run_year: str):
    climate_run_ds = catalog.get_dataset(f'{climate_run_year}-climate-run').to_xarray()
    # downcast to float32
    for var in list(climate_run_ds):
        climate_run_ds[var] = climate_run_ds[var].astype('float32')
    return climate_run_ds


def write_to_icechunk(ds: xr.Dataset, climate_run_year: str):
    storage = icechunk.s3_storage(
        bucket='carbonplan-ocr',
        prefix=f'input/fire-risk/tensor/USFS/{climate_run_year}-climate-run-30m-4326.icechunk',
        region='us-west-2',
    )
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session('main')
    to_icechunk(ds, session, mode='w')
    session.commit('reproject and interp')


def create_unburnable_mask(climate_run_year: str):
    """Create a mask of unburnable areas based on burn probability"""
    climate_run_ds = load_climate_run_ds(climate_run_year).persist()
    unburnable_mask_270 = (climate_run_ds['BP'] == 0).to_dataset(name='unburnable')
    unburnable_mask_270['unburnable'].attrs = {
        'description': f'Mask of unburnable areas based on burn probability from USFS climate run {climate_run_year}',
        'source': 'USFS RDS-2025-0006',
    }
    unburnable_mask_270 = assign_crs(unburnable_mask_270, crs='EPSG:5070')
    unburnable_mask_4326 = xr_reproject(unburnable_mask_270, how='EPSG:4326')
    rps_30_4326 = catalog.get_dataset('USFS-wildfire-risk-communities-4326').to_xarray()
    unburnable_mask_30m_4326 = unburnable_mask_4326.interp_like(
        rps_30_4326, method='nearest', kwargs={'fill_value': True, 'bounds_error': False}
    ).chunk({'latitude': 6000, 'longitude': 4500})
    console.print(unburnable_mask_30m_4326)
    unburnable_mask_30m_4326 = dask.base.optimize(unburnable_mask_30m_4326)[0]
    write_to_icechunk(unburnable_mask_30m_4326, f'unburnable-mask-{climate_run_year}')


def interpolate_and_reproject(climate_run_year: str):
    console.print(f'Processing climate run year: {climate_run_year}')
    climate_run_ds = load_climate_run_ds(climate_run_year).persist()
    console.print(f'Pre-reproject: {climate_run_ds}')

    # assign crs and reproject to lat/lon EPSG:4326
    climate_run_ds = assign_crs(climate_run_ds, crs='EPSG:5070')
    climate_run_4326 = xr_reproject(climate_run_ds, how='EPSG:4326')

    console.print(f'Post-reproject: {climate_run_4326}')

    # load target dataset for interpolation
    rps_30_4326 = catalog.get_dataset('USFS-wildfire-risk-communities-4326').to_xarray()

    # interpolate to 30m using the reprojected target
    climate_run_4326 = climate_run_4326.interp_like(
        rps_30_4326, kwargs={'fill_value': 'extrapolate', 'bounds_error': False}
    )
    # interp_like produces values slightly less then 0, which causes downstream issues. We are clipping to 0 as a min of burn probability.
    climate_run_4326 = climate_run_4326.clip(min=0).chunk({'latitude': 6000, 'longitude': 4500})
    climate_run_4326 = dask.base.optimize(climate_run_4326)[0]

    console.print(f'Post-interpolate: {climate_run_4326}')

    # assign processing attributes
    climate_run_4326.attrs = {
        'title': 'RDS-2025-0006',
        'version': '2025',
        'data_source': 'https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0006',
        'description': 'Modified version of: Spatial datasets of probabilistic wildfire risk components for the conterminous United States for circa 2011 climate and projected future climate circa 2047. This dataset was created by combining from multiple source tif files, interpolating from 270m to 30m and projecting from EPSG:5070 to EPSG:4326. It is stored in the Icechunk storage format.',
        'EPSG': '4326',
        'resolution': '30m',
    }

    console.print(climate_run_4326)
    climate_run_4326 = dask.base.optimize(climate_run_4326)[0]

    # Write to icechunk
    write_to_icechunk(climate_run_4326, climate_run_year)
    console.print(f'Completed processing climate run year: {climate_run_year}')


def main():
    args = {
        'name': 'reproject-and-interpolate-climate-runs',
        'n_workers': 10,
        'region': 'us-west-2',
        'tags': {'Project': 'OCR'},
        'worker_vm_types': 'r7a.24xlarge',
        'scheduler_vm_types': 'r7a.xlarge',
        'spot_policy': 'spot_with_fallback',
        'software': 'ocr-main',
        'wait_for_workers': 1,
        'idle_timeout': '5 minutes',
    }
    console.log(f'Creating Coiled cluster with args: {args}')
    cluster = coiled.Cluster(**args)
    # Touch the client to ensure startup
    client = cluster.get_client()
    console.log(
        f'Cluster {cluster.name} started with {len(client.scheduler_info()["workers"])} workers (initial).'
    )
    try:
        interpolate_and_reproject(climate_run_year='2011')
        interpolate_and_reproject(climate_run_year='2047')
        create_unburnable_mask(climate_run_year='2011')
        create_unburnable_mask(climate_run_year='2047')
    except Exception as e:
        console.log(f'Error occurred: {e}')
        client.cluster.close()
    finally:
        client.cluster.close()
        console.log('Cluster closed.')


if __name__ == '__main__':
    main()
