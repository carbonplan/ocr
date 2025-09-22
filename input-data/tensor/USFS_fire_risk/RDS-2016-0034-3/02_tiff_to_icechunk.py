# Takes the tiff files in the unzipped archive from `01_transfer_src_zip_to_s3.sh` and writes icechunk store(s)
# Note: This can be run locally or on coiled

import coiled
import icechunk
import xarray as xr
from icechunk.xarray import to_icechunk
from odc.geo.xr import assign_crs, xr_reproject

cluster = coiled.Cluster(
    name='ocr_RDS_2016-0034-3',
    region='us-west-2',
    n_workers=[2, 40],
    tags={'Project': 'OCR'},
    worker_vm_types='m8g.large',
    scheduler_vm_types='c8g.4xlarge',
)

client = cluster.get_client()

cluster.adapt(minimum=1, maximum=200)


var_list = ['BP', 'FLP1', 'FLP2', 'FLP3', 'FLP4', 'FLP5', 'FLP6']
fpath_dict = {
    var_name: f's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2016-0032-3/input_tif/CONUS_{var_name}.tif'
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

# assign crs and reproject to lat/lon EPSG:4326
ds = assign_crs(merge_ds, crs='EPSG:5070')
ds_4326 = xr_reproject(ds, how='EPSG:4326')

# assign processing attributes
ds_4326.attrs = {
    'title': 'RDS-2016-0034-3',
    'version': '3rd edition. Published 2023. Downloaded 09-22-25.',
    'data_source': 'https://doi.org/10.2737/RDS-2016-0034-3',
    'description': 'National data on burn probability (BP) and conditional flame-length probability (FLP) were generated for the conterminous United States (CONUS), Alaska, and Hawaii using a geospatial Fire Simulation (FSim) system developed by the USDA Forest Service Missoula Fire Sciences Laboratory. The FSim system includes modules for weather generation, wildfire occurrence, fire growth, and fire suppression. FSim is designed to simulate the occurrence and growth of wildfires under tens of thousands of hypothetical contemporary fire seasons in order to estimate the probability of a given area (i.e., pixel) burning under current (end of 2020) landscape conditions and fire management practices. The data presented here represent modeled BP and FLPs for the United States (US) at a 270-meter grid spatial resolution. Flame-length probability is estimated for six standard Fire Intensity Levels. The six FILs correspond to flame-length classes as follows: FLP1 = < 2 feet (ft); FLP2 = 2 < 4 ft.; FLP3 = 4 < 6 ft.; FLP4 = 6 < 8 ft.; FLP5 = 8 < 12 ft.; FLP6 = 12+ ft. Because they indicate conditional probabilities (i.e., representing the likelihood of burning at a certain intensity level, given that a fire occurs), the FLP data must be used in conjunction with the BP data for risk assessment.',
    'EPSG': '4326',
    'resolution': '270m',
    'DOI': 'https://www.fs.usda.gov/rds/archive/catalog/RDS-2016-0034-3',
}

# chunks to ~100MB
ds_4326 = ds_4326.chunk({'latitude': 3500, 'longitude': 7000})

# Write to icechunk
storage = icechunk.s3_storage(
    bucket='carbonplan-ocr',
    prefix='input/fire-risk/tensor/USFS/RDS-2016-0034-3-epsg_4326.icechunk',
    region='us-west-2',
)
repo = icechunk.Repository.open_or_create(storage)
session = repo.writable_session('main')

to_icechunk(ds_4326, session)
session.commit('Write reprojected Icechunk')
