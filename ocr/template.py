from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask
import icechunk

if TYPE_CHECKING:
    import dask
    import icechunk
    import xarray as xr


def create_template(
    bucket: str = 'carbonplan-ocr',
    prefix: str = 'intermediate/fire-risk/tensor/TEST/atomic_unit_risk_test',
):
    """
    TODO: TO BE DEPRECIATED

    """
    import dask
    import icechunk
    import numpy as np
    import xarray as xr

    from ocr import catalog
    from ocr.chunking_config import ChunkingConfig

    config = ChunkingConfig()

    # use the 30m projected space USFS risk as template
    ds = catalog.get_dataset('USFS-wildfire-risk-communities').to_xarray()[['BP']]
    ds['BP'] = ds['BP'].astype('float32')
    ds['BP'].encoding = {}
    # y, x: 6000, 4500

    bucket = 'carbonplan-ocr'
    prefix = 'intermediate/fire-risk/tensor/TEST/atomic_unit_risk_test'
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session('main')

    template = xr.Dataset(config.ds.coords).drop_vars('spatial_ref')
    template['BP'] = xr.DataArray(
        dask.array.zeros(
            (config.ds.sizes['y'], config.ds.sizes['x']),
            dtype='float32',
            chunks=-1,
        ),
        dims=('y', 'x'),
    )

    template.to_zarr(
        session.store,
        compute=False,
        mode='w',
        encoding={
            'BP': {'chunks': ((config.chunks['x'], config.chunks['y'])), 'fill_value': np.nan}
        },
        consolidated=False,
    )

    session.commit('template')


@dataclass
class TemplateConfig:
    bucket: str = 'carbonplan-ocr'
    prefix: str = 'intermediate/fire-risk/tensor/TEST/TEMPLATE.icechunk'
    uri: str = None

    def init_icechunk_repo(self) -> dict:
        """Creates an icechunk repo.

        Args:
            bucket (str, optional): aws bucket name. Defaults to 'carbonplan-ocr'.
            prefix (str, optional): bucket prefix. Defaults to f'intermediate/fire-risk/tensor/TEST/TEMPLATE.icechunk'.

        """
        storage = icechunk.s3_storage(bucket=self.bucket, prefix=self.prefix, from_env=True)
        icechunk.Repository.create(storage)

    def repo_and_session(self, readonly: bool = False, branch: str = 'main'):
        storage = icechunk.s3_storage(bucket=self.bucket, prefix=self.prefix, from_env=True)
        repo = icechunk.Repository.open(storage)
        if readonly:
            session = repo.readonly_session()
        else:
            session = repo.writable_session(branch=branch)
        return {'repo': repo, 'session': session}

    def wipe_icechunk_repo(self):
        # TODO: It might be nice to be able to wipe the template
        raise NotImplementedError("This functionality isn't added yet.")

    def create_template(self):
        import numpy as np
        import xarray as xr

        from ocr.chunking_config import ChunkingConfig
        # NOTE: This is hardcoded as using the USFS 30m chunking scheme!

        config = ChunkingConfig()

        ds = config.ds
        ds['CRPS'].encoding = {}

        repo_session = self.repo_and_session()
        template = xr.Dataset(ds.coords).drop_vars('spatial_ref')
        var_encoding_dict = {
            'chunks': ((config.chunks['longitude'], config.chunks['latitude'])),
            'fill_value': np.nan,
        }
        template_data_array = xr.DataArray(
            dask.array.zeros(
                (config.ds.sizes['latitude'], config.ds.sizes['longitude']),
                dtype='float32',
                chunks=-1,
            ),
            dims=('latitude', 'longitude'),
        )
        template['wind_risk'] = template_data_array
        template['USFS_RPS'] = template_data_array
        # Should we modify the encoding var name to match the output of wind: 'risk'?
        template.to_zarr(
            repo_session['session'].store,
            compute=False,
            mode='w',
            encoding={'wind_risk': var_encoding_dict, 'USFS_RPS': var_encoding_dict},
            consolidated=False,
        )

        repo_session['session'].commit('template')

    def check_icechunk_ancestry():
        raise NotImplementedError('TODO: Not complete')

    def __post_init__(self):
        # TODO: Make this more robust with cloudpathlib or UPath?
        self.uri = 's3://' + self.bucket + '/' + self.prefix


def get_commit_messages_ancestry(repo: icechunk.repository) -> list:
    return [commit.message for commit in list(repo.ancestry(branch='main'))]


def insert_region_uncoop(subset_ds: xr.Dataset, region_id: str):
    from ocr.template import TemplateConfig

    template_config = TemplateConfig()
    icechunk_repo_and_session = template_config.repo_and_session()

    while True:
        try:
            subset_ds.to_zarr(
                icechunk_repo_and_session['session'].store,
                region='auto',
                consolidated=False,
            )

            icechunk_repo_and_session['session'].commit(f'{region_id}')
            print(f'Wrote region: {region_id}')
            break

        except icechunk.ConflictError:
            print(f'conflict for region_commit_history {region_id}, retrying')
            pass


# @dask.delayed
# def insert_region_uncoop(subset_ds: xr.Dataset, bucket: str, prefix: str, region_id: str):
#     storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)

#     repo = icechunk.Repository.open(storage)

#     while True:
#         try:
#             session = repo.writable_session('main')
#             subset_ds.to_zarr(
#                 session.store,
#                 region='auto',
#                 consolidated=False,
#             )

#             session.commit(f'{region_id}')
#             print(f'Wrote region: {region_id}')
#             break

#         except icechunk.ConflictError:
#             print(f'conflict for region_commit_history {region_id}, retrying')
#             pass


def return_ds_subsets_from_region_dict(ds: xr.Dataset, region_dict: dict) -> dict:
    return {
        region_id: ds.sel(x=x_slice, y=y_slice)
        for region_id, (x_slice, y_slice) in region_dict.items()
    }


def filter_region_duplicates(repo: icechunk.Repository, region_dict) -> dict:
    commit_messages = get_commit_messages_ancestry(repo)
    already_commited_messages = [
        msg
        for message in commit_messages
        for msg in (message.split(',') if ',' in message else [message])
    ]
    uncommited_dict = {
        key: subset for key, subset in region_dict.items() if key not in already_commited_messages
    }
    if not uncommited_dict:
        # ie empty dict
        raise ValueError('No new regions to commit')
    else:
        return uncommited_dict


def write_regions(ds: xr.Dataset, bucket: str, prefix: str, region_dict: dict):
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
    repo = icechunk.Repository.open(storage)
    session = repo.writable_session('main')

    # filter out regions that have already been commited

    uncommited_dict = filter_region_duplicates(repo, region_dict=region_dict)

    # create dataset subsets
    ds_subsets_uncommited = return_ds_subsets_from_region_dict(ds=ds, region_dict=uncommited_dict)
    with session.allow_pickling():
        tasks = [
            insert_region_uncoop(
                subset_ds=subset_ds, bucket=bucket, prefix=prefix, region_id=region_id
            )
            for region_id, subset_ds in ds_subsets_uncommited.items()
        ]
        dask.compute(*tasks)
