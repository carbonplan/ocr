# Takes the tiff files in the unzipped archive from `01_transfer_src_zip_to_s3.sh` and writes icechunk store(s)
# Note: This can be run locally or on coiled

import coiled
import icechunk
import xarray as xr
from icechunk.xarray import to_icechunk


def build_icechunk(ds_name: str):
    var_name_list = [
        'BP.tif',
        'FLP1.tif',
        'FLP2.tif',
        'FLP3.tif',
        'FLP4.tif',
        'FLP5.tif',
        'FLP6.tif',
    ]

    path = f's3://carbonplan-data/USFS/RDS-2025-0006/Data/{ds_name}/'
    url_list = [path + var for var in var_name_list]

    def preprocess(ds, filename):
        return ds.rename({'band_data': filename})

    combined_ds = xr.open_mfdataset(
        url_list,
        combine='by_coords',
        preprocess=lambda ds: preprocess(ds, ds.encoding['source'].split('/')[-1].split('.')[0]),
        engine='rasterio',
        parallel=True,
    )
    # remove 1d band dim and chunk to ~100MB
    combined_ds = combined_ds.squeeze().drop_vars('band').chunk({'y': 6000, 'x': 4500})

    storage = icechunk.s3_storage(
        bucket='carbonplan-data',
        prefix=f'USFS/RDS-2025-0006/{ds_name}_Icechunk',
        region='us-west-2',
    )
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session('main')
    to_icechunk(combined_ds, session)
    first_snapshot = session.commit('Add all raster data to store')
    print(first_snapshot)


@coiled.function(region='us-west-2', vm_type='m8g.large')
def main():
    build_icechunk('2011ClimateRun')
    build_icechunk('2047ClimateRun')


if __name__ == '__main__':
    main()
