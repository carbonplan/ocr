"""Risk-agnostic storage config. Fire still uses its own config! TODO: refactor fire config to this later"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import icechunk
import pydantic_settings
from obstore.auth.boto3 import Boto3CredentialProvider
from obstore.store import LocalStore, S3Store
from pydantic import BeforeValidator, field_validator
from pydantic.dataclasses import dataclass
from pydantic_extra_types.semantic_version import SemanticVersion


class StorageConfig(pydantic_settings.BaseSettings):
    # we can override, just a default
    bucket: str = 'carbonplan-ocr'
    region: str = 'us-west-2'
    local_root: Path | None = None

    # Fields are read from env vars OCR_STORAGE_BUCKET, OCR_STORAGE_REGION, OCR_STORAGE_LOCAL_ROOT.
    model_config = pydantic_settings.SettingsConfigDict(env_prefix='OCR_STORAGE_')

    @field_validator('local_root')
    @classmethod
    def _expand(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v else v

    def object_store(self) -> LocalStore | S3Store:
        if self.local_root:
            return LocalStore(self.local_root, mkdir=True)
        # boto3's chain covers SSO profiles, which obstore's own (env/IMDS only) misses on laptops.
        return S3Store(
            self.bucket, region=self.region, credential_provider=Boto3CredentialProvider()
        )

    def uri(self, key: str) -> str:
        if self.local_root:
            return str(self.local_root / key)
        return f's3://{self.bucket}/{key}'

    def icechunk_storage(self, prefix: str) -> icechunk.Storage:
        if self.local_root:
            return icechunk.local_filesystem_storage(str(self.local_root / prefix))
        return icechunk.s3_storage(bucket=self.bucket, prefix=prefix, region=self.region)


class Risk(StrEnum):
    FIRE = 'fire'
    FLOOD = 'flood'
    HURRICANE = 'hurricane'


def _strip_v(version: str) -> str:
    if not version.startswith('v'):
        raise ValueError(f'version must start with v (e.g. v2.0.0), got {version!r}')
    return version[1:]


@dataclass(frozen=True)
class OutputPaths:
    """Bucket-relative output keys for one risk version. Resolve with StorageConfig.uri / icechunk_storage.

    >>> cfg, out = StorageConfig(), OutputPaths(Risk.FLOOD, 'v2.0.0')
    >>> repo = icechunk.Repository.open(cfg.icechunk_storage(out.store))
    >>> write_geoparquet(tbl, cfg.uri(out.buildings))
    """

    risk: Risk
    # not a standard str, so we can validate as semver
    version: Annotated[SemanticVersion, BeforeValidator(_strip_v)]

    @property
    def root(self) -> str:
        return f'output/{self.risk}/v{self.version}'

    @property
    def store(self) -> str:
        return f'{self.root}/store.icechunk'

    @property
    def buildings(self) -> str:
        return f'{self.root}/vector/buildings.geoparquet'

    @property
    def regions(self) -> str:
        """Prefix for region files (nation, states, per-state counties/tracts/blocks)."""
        return f'{self.root}/vector/regions'
