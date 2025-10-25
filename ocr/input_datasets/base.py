"""Base classes and utilities for input dataset processing."""

import abc
import tempfile
from pathlib import Path
from typing import Literal

import pooch
import pydantic
import pydantic_settings

from ocr.console import console


class InputDatasetConfig(pydantic_settings.BaseSettings):
    """Configuration for input dataset processing."""

    s3_bucket: str = pydantic.Field(
        default='carbonplan-ocr', description='S3 bucket for storing processed datasets'
    )
    s3_region: str = pydantic.Field(default='us-west-2', description='AWS region for S3 operations')
    base_prefix: str = pydantic.Field(
        default='input/fire-risk', description='Base S3 prefix for input datasets'
    )
    temp_dir: Path | None = pydantic.Field(
        default=None, description='Temporary directory for downloads. Uses system temp if None'
    )
    chunk_size: int = pydantic.Field(
        default=8192, description='Chunk size in bytes for streaming downloads'
    )
    debug: bool = pydantic.Field(default=False, description='Enable debug logging')

    model_config = {'env_prefix': 'ocr_input_dataset_', 'case_sensitive': False}

    def model_post_init(self, __context):
        """Set default temp directory if not provided."""
        if self.temp_dir is None:
            object.__setattr__(self, 'temp_dir', Path(tempfile.gettempdir()))


class BaseDatasetProcessor(abc.ABC):
    """Abstract base class for input dataset processors.

    Each dataset processor handles downloading, processing, and uploading
    a specific input dataset to S3/Icechunk storage.
    """

    # Dataset metadata (override in subclasses)
    dataset_name: str = ''
    dataset_type: Literal['tensor', 'vector'] = 'tensor'
    description: str = ''
    source_url: str = ''
    version: str = ''

    def __init__(self, config: InputDatasetConfig | None = None, *, dry_run: bool = False):
        """Initialize the dataset processor.

        Parameters
        ----------
        config : InputDatasetConfig, optional
            Configuration object. Creates default if None.
        dry_run : bool, default False
            If True, skip actual S3 writes and only log operations.
        """
        self.config = config if config is not None else InputDatasetConfig()
        self.dry_run = dry_run
        self._temp_dir: Path | None = None

    @property
    def temp_dir(self) -> Path:
        """Get or create temporary directory for this processor."""
        if self._temp_dir is None:
            temp_base = (
                self.config.temp_dir if self.config.temp_dir else Path(tempfile.gettempdir())
            )
            self._temp_dir = temp_base / self.dataset_name
            self._temp_dir.mkdir(parents=True, exist_ok=True)
        return self._temp_dir

    def retrieve(
        self,
        url: str,
        known_hash: str | None = None,
        *,
        fname: str | None = None,
        processor: pooch.processors.ExtractorProcessor | None = None,
    ) -> str | list[str]:
        """Download and cache a file using pooch.

        This method uses pooch for robust downloading with hash verification,
        automatic caching, and optional post-processing (unzipping, etc.).

        Parameters
        ----------
        url : str
            URL to download from. Supports HTTP, FTP, and DOI (e.g., doi:10.5281/zenodo.1234).
        known_hash : str, optional
            Expected hash of the file for verification (format: 'md5:abc123' or 'sha256:def456').
            If None, hash verification is skipped (not recommended for production).
        fname : str, optional
            File name to use for the cached file. If None, extracts from URL.
        processor : pooch.processors.ExtractorProcessor, optional
            Post-processor to apply after download (e.g., pooch.Unzip() or pooch.Untar()).

        Returns
        -------
        str or list[str]
            Path to the downloaded file. If a processor is used, returns list of extracted files.

        Examples
        --------
        Download a single file:
        >>> path = self.retrieve(
        ...     url='https://example.com/data.csv',
        ...     known_hash='md5:abc123',
        ... )

        Download and unzip:
        >>> files = self.retrieve(
        ...     url='https://example.com/data.zip',
        ...     known_hash='md5:abc123',
        ...     processor=pooch.Unzip(),
        ... )
        """
        if self.dry_run:
            console.log(f'[DRY RUN] Would retrieve {url} (hash: {known_hash})')
            return str(self.temp_dir / (fname or 'dummy.file'))

        # Use the temp_dir for this dataset as the cache
        cache_path = self.temp_dir
        cache_path.mkdir(parents=True, exist_ok=True)

        # Download and cache the file
        try:
            result = pooch.retrieve(
                url=url,
                known_hash=known_hash,
                fname=fname,
                path=cache_path,
                processor=processor,
                progressbar=True,
            )

            if isinstance(result, list):
                console.log(f'Retrieved and extracted {len(result)} files from {url}')
            else:
                file_size = Path(result).stat().st_size
                console.log(f'Retrieved {Path(result).name} ({file_size / 1e6:.1f} MB)')

            return result
        except Exception as e:
            console.log(f'[bold red]Failed to retrieve {url}: {e}[/bold red]')
            raise

    @abc.abstractmethod
    def download(self) -> None:
        """Download raw source data.

        This method should handle downloading the raw dataset from its source
        (e.g., USFS Box, Census TIGER, etc.) to local temporary storage.
        """
        ...

    @abc.abstractmethod
    def process(self) -> None:
        """Process and upload data to S3/Icechunk.

        This method should handle the main processing pipeline: converting
        formats (TIFF→Icechunk, CSV→Parquet, etc.), reprojecting if needed,
        and uploading to S3 storage.
        """
        ...

    def run_all(self) -> None:
        """Run the complete pipeline: download, process, cleanup."""
        try:
            console.log(f'Starting {self.dataset_name} dataset processing')
            self.download()
            self.process()
            console.log(f'Completed {self.dataset_name} dataset processing')
        finally:
            console.log('Exiting processing...')
