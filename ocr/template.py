from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask
import icechunk

if TYPE_CHECKING:
    import dask
    import icechunk
    import xarray as xr


# ---------------------------------------------------------------------------
# Vector Config
# ---------------------------------------------------------------------------
@dataclass
class VectorConfig:
    branch: str = 'QA'
    bucket: str = 'carbonplan-ocr'
    prefix: str = None
    wipe: bool = False
    # these are defined in post_init since they depend on branch.
    region_geoparquet_prefix: str = None
    region_geoparquet_uri: str = None

    consolidated_geoparquet_prefix: str = None
    consolidated_geoparquet_uri: str = None

    pmtiles_prefix: str = None
    pmtiles_prefix_uri: str = None

    def delete_region_gpqs(self):
        import boto3

        if 'geoparquet_regions' not in self.region_geoparquet_prefix:
            raise ValueError(
                'It seems like the prefix specified is not the region_id tagged geoparq files. [safety switch]'
            )
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.bucket)
        bucket.objects.filter(Prefix=self.region_geoparquet_prefix).delete()

    def _gen_prefixes(self):
        self.region_geoparquet_prefix = self.prefix + 'geoparquet_regions/'
        self.consolidated_geoparquet_prefix = self.prefix + 'consolidated_geoparquet.parquet'
        self.pmtiles_prefix = self.prefix + 'consolidated.pmtiles'

    def _gen_uris(self):
        # TODO: Make this more robust with cloudpathlib or UPath
        self.region_geoparquet_uri = 's3://' + self.bucket + '/' + self.region_geoparquet_prefix
        self.consolidated_geoparquet_uri = (
            's3://' + self.bucket + '/' + self.consolidated_geoparquet_prefix
        )
        self.pmtiles_prefix_uri = 's3://' + self.bucket + '/' + self.pmtiles_prefix

    def __post_init__(self):
        self.region_geoparquet_prefix: str = None
        self.consolidated_geoparquet_prefix: str = None
        self.pmtiles_prefix: str = None
        if self.branch == 'prod':
            self.prefix = 'intermediate/fire-risk/vector/prod/'
        elif self.branch == 'QA':
            self.prefix = 'intermediate/fire-risk/vector/QA/'
        else:
            raise ValueError(f'{self.branch} is not a valid branch. Valid options are: [QA, prod]')
        if self.wipe:
            self._gen_prefixes()
            self._gen_uris()
            self.delete_region_gpqs()

        self._gen_prefixes()
        self._gen_uris()


# ---------------------------------------------------------------------------
# Pyramid Config
# ---------------------------------------------------------------------------


@dataclass
class PyramidConfig:
    branch: str = 'QA'
    bucket: str = 'carbonplan-ocr'
    prefix: str = None
    uri: str = None

    def __post_init__(self):
        if self.branch == 'prod':
            self.prefix = 'intermediate/fire-risk/tensor/prod/pyramid.zarr'
        elif self.branch == 'QA':
            self.prefix = 'intermediate/fire-risk/tensor/QA/pyramid.zarr'
        else:
            raise ValueError(f'{self.branch} is not a valid branch. Valid options are: [QA, prod]')

        # TODO: Make this more robust with cloudpathlib or UPath
        self.uri = 's3://' + self.bucket + '/' + self.prefix


# ---------------------------------------------------------------------------
# Icechunk Template Config
# ---------------------------------------------------------------------------
@dataclass
class IcechunkConfig:
    branch: str = 'QA'
    bucket: str = 'carbonplan-ocr'
    prefix: str = None
    wipe: bool = False
    uri: str = None

    def init_icechunk_repo(self) -> dict:
        """Creates an icechunk repo or opens if does not exist

        Args:
            bucket (str, optional): aws bucket name. Defaults to 'carbonplan-ocr'.


        """
        storage = icechunk.s3_storage(bucket=self.bucket, prefix=self.prefix, from_env=True)
        icechunk.Repository.open_or_create(storage)

    def repo_and_session(self, readonly: bool = False, branch: str = 'main'):
        storage = icechunk.s3_storage(bucket=self.bucket, prefix=self.prefix, from_env=True)
        repo = icechunk.Repository.open(storage)
        if readonly:
            session = repo.readonly_session(branch=branch)
        else:
            session = repo.writable_session(branch=branch)
        return {'repo': repo, 'session': session}

    def delete_icechunk_repo(self):
        # Note: Be careful! Big danger!
        import boto3

        if 'template.icechunk' not in self.prefix:
            raise ValueError(
                'It seems like the prefix specified is not the icechunk template. [safety switch]'
            )

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.bucket)
        bucket.objects.filter(Prefix=self.prefix).delete()

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
        template['risk_2011'] = template_data_array
        template['risk_2047'] = template_data_array
        template['wind_risk_2011'] = template_data_array
        template['wind_risk_2047'] = template_data_array
        vars = ['risk_2011', 'risk_2047', 'wind_risk_2011', 'wind_risk_2047']
        # Should we modify the encoding var name to match the output of wind: 'risk'?
        template_encoding_dict = {var: var_encoding_dict for var in vars}
        template.to_zarr(
            repo_session['session'].store,
            compute=False,
            mode='w',
            encoding=template_encoding_dict,
            consolidated=False,
        )

        repo_session['session'].commit('template')

    def __post_init__(self):
        if self.branch == 'prod':
            self.prefix = 'intermediate/fire-risk/tensor/prod/template.icechunk'
        elif self.branch == 'QA':
            # add logging that QA branch cleared the existing repo and re-inited.
            self.prefix = 'intermediate/fire-risk/tensor/QA/template.icechunk'
        else:
            # this is shared, so we could make it a type
            raise ValueError(f'{self.branch} is not a valid branch. Valid options are: [QA, prod]')

        if self.wipe:
            # add logging that wipe flag cleared the existing repo and re-inited.
            self.delete_icechunk_repo()
            self.init_icechunk_repo()
            self.create_template()

        # TODO: Make this more robust with cloudpathlib or UPath
        self.uri = 's3://' + self.bucket + '/' + self.prefix


def get_commit_messages_ancestry(repo: icechunk.repository, icechunk_branch: str = 'main') -> list:
    commit_messages = [commit.message for commit in list(repo.ancestry(branch=icechunk_branch))]
    # separate commits by ',' and handle case of single length ancestry commit history
    split_commits = [
        msg
        for message in commit_messages
        for msg in (message.split(',') if ',' in message else [message])
    ]
    return split_commits


def region_id_exists_in_repo(region_id: str, branch: str):
    from ocr.template import IcechunkConfig

    icechunk_config = IcechunkConfig(branch=branch)
    icechunk_repo_and_session = icechunk_config.repo_and_session()

    region_ids_in_ancestry = get_commit_messages_ancestry(icechunk_repo_and_session['repo'])

    # check if region_id is already in icechunk commit history
    if region_id in region_ids_in_ancestry:
        return True
    else:
        # switch to logging
        print(f'{region_id} already exists in Icechunk ancestry')
        return False


def insert_region_uncoop(subset_ds: xr.Dataset, region_id: str, branch: str, wipe: bool = False):
    import icechunk

    from ocr.template import IcechunkConfig

    icechunk_config = IcechunkConfig(branch=branch, wipe=wipe)
    icechunk_repo_and_session = icechunk_config.repo_and_session()

    while True:
        try:
            subset_ds.to_zarr(
                icechunk_repo_and_session['session'].store,
                region='auto',
                consolidated=False,
            )
            # Trying out the rebase strategy described here: https://github.com/earth-mover/icechunk/discussions/802#discussioncomment-13064039
            # We should be in the same position, where we don't have real conflicts, just write timing conflicts.
            icechunk_repo_and_session['session'].commit(
                f'{region_id}', rebase_with=icechunk.ConflictDetector()
            )
            print(f'Wrote region: {region_id}')
            break

        except icechunk.ConflictError:
            # add logging
            print(f'conflict for region_commit_history {region_id}, retrying')
            pass
