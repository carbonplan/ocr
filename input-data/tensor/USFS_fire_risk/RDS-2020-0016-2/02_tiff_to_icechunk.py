# Takes the tiff files in the unzipped archive from `01_transfer_src_zip_to_s3.sh` and writes icechunk store(s)
# Note: This can be run locally or on coiled

import coiled
import icechunk
import xarray as xr
from icechunk.xarray import to_icechunk

cluster = coiled.Cluster(
    name='ocr_RDS_2020-0016-02',
    region='us-west-2',
    n_workers=10,
    tags={'project': 'OCR'},
    worker_vm_types='m8g.large',
    scheduler_vm_types='m8g.2xlarge',
)

client = cluster.get_client()

cluster.adapt(minimum=1, maximum=200)


var_list = ['BP', 'CRPS', 'CFL', 'Exposure', 'FLEP4', 'FLEP8', 'WHP']
fpath_dict = {
    var_name: f's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/{var_name}_CONUS.tif'
    for var_name in var_list
}

# merge all the datasets and rename vars
merge_ds = xr.merge(
    [
        xr.open_dataset(fpath, engine='rasterio')
        .squeeze()
        .drop_vars('band')
        .rename({'band_data': var_name})
        for var_name, fpath in fpath_dict.items()
    ],
    compat='override',
    join='override',
)

merge_ds = merge_ds.chunk({'y': 6000, 'x': 4500})

storage = icechunk.s3_storage(
    bucket='carbonplan-ocr',
    prefix='input/fire-risk/tensor/USFS/RDS-2022-0016-02_all_vars_merge_icechunk',
    region='us-west-2',
)
repo = icechunk.Repository.open_or_create(storage)

session = repo.writable_session('main')
to_icechunk(merge_ds, session)
snapshot = session.commit('Add all variables to store')
