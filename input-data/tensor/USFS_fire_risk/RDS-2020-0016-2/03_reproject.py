# COILED n-tasks 1
# COILED --region us-west-2
# COILED --vm-type m7i.8xlarge
# COILED --forward-aws-credentials
# COILED --tag Project=OCR


# Script to reProject the USFS 30m Community Risk dataset to EPSG:4326


import icechunk
import xarray as xr
from icechunk.xarray import to_icechunk
from odc.geo.xr import assign_crs, xr_reproject

from ocr import catalog


def load_USFS_community_risk() -> xr.Dataset:
    rps_30 = catalog.get_dataset(
        'USFS-wildfire-risk-communities'
    ).to_xarray()  # downcast to float32
    for var in list(rps_30):
        rps_30[var] = rps_30[var].astype('float32')
    return rps_30


def write_to_icechunk(ds: xr.Dataset):
    storage = icechunk.s3_storage(
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/USFS/RDS-2022-0016-02_EPSG_4326_icechunk_all_vars',
        region='us-west-2',
    )
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session('main')
    to_icechunk(ds, session)
    session.commit('reproject to 4326')


def interpolate_and_reproject():
    rps_30 = load_USFS_community_risk()

    # assign crs and reproject to lat/lon EPSG:4326
    rps_30 = assign_crs(rps_30, crs='EPSG:5070')
    rps_30_4326 = xr_reproject(rps_30, how='EPSG:4326')

    # assign processing attributes
    rps_30.attrs = {
        'title': 'RDS-2020-0016-02',
        'version': '2024-V2',
        'data_source': 'https://www.fs.usda.gov/rds/archive/catalog/RDS-2020-0016-2',
        'description': 'Modified version of: Wildfire Risk to Communities: Spatial datasets of landscape-wide wildfire risk components for the United States (2nd Edition). This dataset was created by combining from multiple source tif files and re-projecting from EPSG:5070 to EPSG:4326. It is stored in the Icechunk storage format.',
        'EPSG': '4326',
        'resolution': '30m',
    }

    # Write to icechunk
    write_to_icechunk(rps_30_4326)


def main():
    interpolate_and_reproject()


if __name__ == '__main__':
    main()
