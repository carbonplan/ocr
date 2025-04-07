# Takes the tiff files in the unzipped archive from `01_transfer_src_zip_to_s3.sh` and writes icechunk store(s)
# Note: This can be run locally or on coiled

import coiled
import icechunk
import xarray as xr
from icechunk.xarray import to_icechunk


def build_icechunk():
    path = 's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/BP_CONUS.tif'
    ds = xr.open_dataset(path, engine='rasterio', chunks={})
    # remove 1d band dim and chunk to ~100MB
    ds = ds.squeeze().drop_vars('band').chunk({'y': 6000, 'x': 4500})

    storage = icechunk.s3_storage(
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/USFS/RDS-2022-0016-02_Icechunk',
        region='us-west-2',
    )
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session('main')
    to_icechunk(ds, session)
    snapshot = session.commit('Add BP raster to store')
    print(snapshot)


@coiled.function(
    region='us-west-2', n_workers=[1, 20], vm_type='m8g.large', tags={'Project': 'OCR'}
)
def main():
    build_icechunk()


if __name__ == '__main__':
    main()
