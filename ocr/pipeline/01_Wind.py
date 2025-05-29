# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m7a.xlarge
# COILED --tag project=OCR


# Note!: This is a placeholder for the wind informed calc.
# We are using a subset / atomic unit from the 30m USFS risk and writing it to an intermediate product.

import icechunk
import numpy as np

from ocr import catalog

#  should we use distributed on a single AU w/ a small VM?

region_id = 'y10_x2'
# hardcoding for now:
# y_slice, x_slice = chunk_slices[region_id]
# plot_subset_ds(ds.isel(y=y_slice,x=x_slice))


y_slice = slice(np.int64(60000), np.int64(66000), None)
x_slice = slice(np.int64(9000), np.int64(13500), None)

ds = catalog.get_dataset('USFS-wildfire-risk-communities').to_xarray()[['BP']]
ds['BP'] = ds['BP'].astype('float32')

subset_ds = ds.isel(y=y_slice, x=x_slice)


bucket = 'carbonplan-ocr'
prefix = f'intermediate/fire-risk/tensor/TEST/BP_{region_id}'
storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
repo = icechunk.Repository.open_or_create(storage)
session = repo.writable_session('main')

subset_ds.to_zarr(
    session.store,
    mode='w',
    consolidated=False,
)

session.commit('init')
# eventually this should be region_id?
