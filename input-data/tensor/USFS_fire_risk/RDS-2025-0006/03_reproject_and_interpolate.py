# Script to modify the 2011 and 2014 USFS climate run datasets:
# - Reproject from EPSG:5070 to EPSG:4326 (X/Y -> lat/lon)
# - Interpolate from 270m to 30m
# - Note: This uses the USFS Community Risk 30m dataset as an input for interpolation


import coiled
import icechunk
import xarray as xr
from icechunk.xarray import to_icechunk
from odc.geo.xr import assign_crs, xr_reproject

from ocr import catalog


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
    to_icechunk(ds, session)
    session.commit('reproject and interp')


def interpolate_and_reproject(climate_run_year: str):
    climate_run_ds = load_climate_run_ds(climate_run_year).persist()
    rps_30 = (
        catalog.get_dataset('USFS-wildfire-risk-communities')
        .to_xarray()['BP']
        .astype('float32')
        .to_dataset()
    )
    # interpolate to 30m & chunk
    interp_30 = climate_run_ds.interp_like(
        rps_30, kwargs={'fill_value': 'extrapolate', 'bounds_error': False}
    )
    # interp_like produces values slightly less then 0, which causes downstream issues. We are clipping to 0 as a min of burn probability.
    interp_30 = interp_30.clip(min=0)

    # assign crs and reproject to lat/lon EPSG:4326
    interp_30 = assign_crs(interp_30, crs='EPSG:5070')

    climate_run_4326 = xr_reproject(interp_30, how='EPSG:4326', chunks=(6000, 4500))

    # ensure the coords match the rps_30_4326 coords exactly
    rps_30_4326 = catalog.get_dataset('USFS-wildfire-risk-communities-4326').to_xarray()
    climate_run_4326 = climate_run_4326.assign_coords(
        latitude=rps_30_4326.latitude, longitude=rps_30_4326.longitude
    )

    # assign processing attributes
    climate_run_4326.attrs = {
        'title': 'RDS-2025-0006',
        'version': '2025',
        'data_source': 'https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0006',
        'description': 'Modified version of: Spatial datasets of probabilistic wildfire risk components for the conterminous United States for circa 2011 climate and projected future climate circa 2047. This dataset was created by combining from multiple source tif files, interpolating from 270m to 30m and projecting from EPSG:5070 to EPSG:4326. It is stored in the Icechunk storage format.',
        'EPSG': '4326',
        'resolution': '30m',
    }

    print(climate_run_4326)

    # Write to icechunk
    write_to_icechunk(climate_run_4326, climate_run_year)


@coiled.function(
    region='us-west-2',
    n_workers=20,
    vm_type='r7g.16xlarge',
    tags={'Project': 'OCR'},
    extra_kwargs=dict(scheduler_vm_types='m8g.4xlarge', wait_for_workers=1),
    software='ocr-main',
)
def main():
    interpolate_and_reproject(climate_run_year='2011')
    interpolate_and_reproject(climate_run_year='2047')


if __name__ == '__main__':
    main()
