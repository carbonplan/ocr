import icechunk
import pydantic
import pydantic_settings
from upath import UPath

from ocr.chunking_config import ChunkingConfig
from ocr.console import console
from ocr.icechunk_utils import get_commit_messages_ancestry
from ocr.types import Branch


class VectorConfig(pydantic_settings.BaseSettings):
    """Configuration for vector data processing."""

    branch: Branch = pydantic.Field(default=Branch.QA, description='Branch for vector processing')
    storage_root: str = pydantic.Field(
        ..., description='Root storage path for vector data, can be a bucket name or local path'
    )
    prefix: str | None = pydantic.Field(None, description='Sub-path within the storage root')
    wipe: bool = pydantic.Field(
        default=False, description='Whether to wipe existing data before processing'
    )

    def model_post_init(self, __context):
        """Post-initialization to set up prefixes and URIs based on branch."""
        if self.prefix is None:
            if self.branch == Branch.PROD:
                self.prefix = 'intermediate/fire-risk/vector/prod/'
            elif self.branch == Branch.QA:
                self.prefix = 'intermediate/fire-risk/vector/QA/'

        if self.wipe:
            self.delete_region_gpqs()

    @property
    def region_geoparquet_prefix(self) -> str:
        return f'{self.prefix}geoparquet-regions/'

    @property
    def region_geoparquet_uri(self) -> UPath:
        return UPath(f'{self.storage_root}/{self.region_geoparquet_prefix}')

    @property
    def consolidated_geoparquet_prefix(self) -> str:
        return f'{self.prefix}consolidated-geoparquet.parquet'

    @property
    def consolidated_geoparquet_uri(self) -> UPath:
        return UPath(f'{self.storage_root}/{self.consolidated_geoparquet_prefix}')

    @property
    def pmtiles_prefix(self) -> str:
        return f'{self.prefix}consolidated.pmtiles'

    @property
    def pmtiles_prefix_uri(self) -> UPath:
        return UPath(f'{self.storage_root}/{self.pmtiles_prefix}')

    @property
    def aggregated_regions_prefix(self) -> UPath:
        return UPath(f'{self.storage_root}/{self.prefix}aggregated-regions/')

    def delete_region_gpqs(self):
        """Delete region geoparquet files from the storage."""
        console.log(f'Deleting region geoparquet files from {self.region_geoparquet_uri}')
        if self.region_geoparquet_prefix is None:
            raise ValueError('Region geoparquet prefix must be set before deletion.')
        if 'geoparquet-regions' not in self.region_geoparquet_prefix:
            raise ValueError(
                'It seems like the prefix specified is not the region_id tagged geoparq files. [safety switch]'
            )

        # Use UPath to handle deletion in a cloud-agnostic way
        # First, get a list of all files in the region geoparquet prefix
        region_path = UPath(self.region_geoparquet_uri)
        if region_path.exists():
            for file in region_path.glob('*'):
                if file.is_file():
                    file.unlink()
        else:
            console.log('No files found to delete.')


class IcechunkConfig(pydantic_settings.BaseSettings):
    """Configuration for icechunk processing."""

    branch: Branch = pydantic.Field(default=Branch.QA, description='Branch for icechunk processing')
    storage_root: str = pydantic.Field(
        ..., description='Root storage path for icechunk data, can be a bucket name or local path'
    )
    prefix: str | None = pydantic.Field(None, description='Sub-path within the storage root')
    wipe: bool = pydantic.Field(
        default=False, description='Whether to wipe existing data before processing'
    )

    def model_post_init(self, __context):
        """Post-initialization to set up prefixes and URIs based on branch."""
        if self.prefix is None:
            if self.branch == Branch.PROD:
                self.prefix = 'intermediate/fire-risk/tensor/prod/template.icechunk'
            elif self.branch == Branch.QA:
                self.prefix = 'intermediate/fire-risk/tensor/QA/template.icechunk'

        if self.wipe:
            self.delete()
            self.init_repo()
            self.create_template()

        commits = get_commit_messages_ancestry(self.repo_and_session()['repo'])
        if 'initialize store with template' not in commits:
            console.log('No template found in icechunk store. Creating a new template dataset.')
            self.create_template()

    @property
    def uri(self) -> UPath:
        """Return the URI for the icechunk repository."""
        if self.prefix is None:
            raise ValueError('Prefix must be set before initializing the icechunk repo.')
        return UPath(f'{self.storage_root}/{self.prefix}')

    @property
    def storage(self) -> icechunk.Storage:
        if self.uri is None:
            raise ValueError('URI must be set before initializing the icechunk repo.')

        protocol = self.uri.protocol
        if protocol == 's3':
            parts = self.uri.parts
            bucket = parts[0].strip('/')
            prefix = '/'.join(parts[1:])
            storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
        elif protocol in {'file', 'local'} or protocol == '':
            storage = icechunk.local_filesystem_storage(path=str(self.uri.path))

        else:
            raise ValueError(
                f'Unsupported protocol: {protocol}. Supported protocols are: [s3, file, local]'
            )
        return storage

    def init_repo(self):
        """Creates an icechunk repo or opens if does not exist"""

        icechunk.Repository.open_or_create(self.storage)
        console.log('Initialized/Opened icechunk repository')

    def repo_and_session(self, readonly: bool = False, branch: str = 'main'):
        """Open an icechunk repository and return the session."""
        storage = self.storage
        repo = icechunk.Repository.open_or_create(storage)
        if readonly:
            session = repo.readonly_session(branch=branch)
        else:
            session = repo.writable_session(branch=branch)

        console.log(
            f'Opened icechunk repository at {self.uri} with branch {branch} in {"readonly" if readonly else "writable"} mode.'
        )
        return {'repo': repo, 'session': session}

    def delete(self):
        """Delete the icechunk repository."""
        if self.uri is None:
            raise ValueError('URI must be set before deleting the icechunk repo.')

        console.log(f'Deleting icechunk repository at {self.uri}')
        if self.uri.protocol == 's3':
            if self.uri.exists():
                for file in self.uri.glob('*'):
                    if file.is_file():
                        file.unlink()
                self.uri.rmdir()
            else:
                console.log('No files found to delete.')

        elif self.uri.protocol in {'file', 'local'} or self.uri.protocol == '':
            path = self.uri.path
            import shutil

            if UPath(path).exists():
                shutil.rmtree(path)
            else:
                console.log('No files found to delete.')

        console.log('Deleted icechunk repository')

    def create_template(self):
        """Create a template dataset for icechunk store"""
        import dask
        import dask.array
        import numpy as np
        import xarray as xr

        repo_and_session = self.repo_and_session()
        # NOTE: This is hardcoded as using the USFS 30m chunking scheme!
        config = ChunkingConfig()

        ds = config.ds
        ds['CRPS'].encoding = {}

        template = xr.Dataset(ds.coords).drop_vars('spatial_ref')
        var_encoding_dict = {
            'chunks': ((config.chunks['longitude'], config.chunks['latitude'])),
            'fill_value': np.nan,
        }
        template_data_array = xr.DataArray(
            dask.array.empty(
                (config.ds.sizes['latitude'], config.ds.sizes['longitude']),
                dtype='float32',
                chunks=-1,
            ),
            dims=('latitude', 'longitude'),
        )
        variables = ['risk_2011', 'risk_2047', 'wind_risk_2011', 'wind_risk_2047']
        template_encoding_dict = {}
        for variable in variables:
            template[variable] = template_data_array
            template_encoding_dict[variable] = var_encoding_dict
        template.to_zarr(
            repo_and_session['session'].store,
            compute=False,
            mode='w',
            encoding=template_encoding_dict,
            consolidated=False,
        )
        repo_and_session['session'].commit('initialize store with template')
        console.log('Created icechunk template')


class OCRConfig(pydantic_settings.BaseSettings):
    """Configuration settings for OCR processing."""

    branch: Branch = pydantic.Field(default=Branch.QA, description='Branch for OCR processing')
    storage_root: str = pydantic.Field(
        ..., description='Root storage path for OCR data, can be a bucket name or local path'
    )
    wipe: bool = pydantic.Field(
        default=False, description='Whether to wipe existing data before processing'
    )
    vector: VectorConfig | None = pydantic.Field(None, description='Vector configuration')
    icechunk: IcechunkConfig | None = pydantic.Field(None, description='Icechunk configuration')
    chunking: ChunkingConfig | None = pydantic.Field(
        None, description='Chunking configuration for OCR processing'
    )

    class Config:
        """Configuration for Pydantic settings."""

        env_prefix = 'ocr_'
        case_sensitive = False

    def model_post_init(self, __context):
        # Pass branch and wipe to VectorConfig if not already set
        if self.vector is None:
            object.__setattr__(
                self,
                'vector',
                VectorConfig(storage_root=self.storage_root, branch=self.branch, wipe=self.wipe),
            )
        if self.icechunk is None:
            object.__setattr__(
                self,
                'icechunk',
                IcechunkConfig(
                    storage_root=self.storage_root,
                    branch=self.branch,
                    wipe=self.wipe,
                ),
            )
        if self.chunking is None:
            object.__setattr__(
                self,
                'chunking',
                ChunkingConfig(),
            )
