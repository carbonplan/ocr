"""Overture Maps building and address data for CONUS.

This module handles downloading and subsetting Overture Maps data:
1. Query buildings and addresses for CONUS bbox from S3
2. Write subsetted data to carbonplan-ocr S3 bucket

Reference:
https://docs.overturemaps.org/
"""

from typing import ClassVar, Literal

import coiled

from ocr.console import console
from ocr.input_datasets.base import BaseDatasetProcessor, InputDatasetConfig


class OvertureProcessor(BaseDatasetProcessor):
    """Processor for Overture Maps building and address data."""

    dataset_name: str = 'overture-maps'
    dataset_type = 'vector'
    description: str = 'Overture Maps building and address data for CONUS'
    source_url: str = 'https://overturemaps.org'
    version: str = '2025-09-24.0'

    # CONUS bounding box (minx, miny, maxx, maxy)
    CONUS_BBOX: ClassVar[tuple[float, float, float, float]] = (
        -125.354004,
        24.413323,
        -66.555176,
        49.196737,
    )

    # Overture S3 bucket and region
    OVERTURE_BUCKET: ClassVar[str] = 'overturemaps-us-west-2'
    OVERTURE_REGION: ClassVar[str] = 'us-west-2'

    # Coiled configuration
    COILED_WORKER_VM: ClassVar[str] = 'c8g.2xlarge'
    COILED_SCHEDULER_VM: ClassVar[str] = 'm8g.xlarge'
    COILED_WORKERS: ClassVar[int] = 1
    COILED_SOFTWARE: ClassVar[str | None] = None

    def __init__(
        self,
        config: InputDatasetConfig | None = None,
        *,
        dry_run: bool = False,
        data_type: Literal['buildings', 'addresses', 'both'] = 'both',
        use_coiled: bool = False,
    ):
        """Initialize the Overture processor.

        Parameters
        ----------
        config : InputDatasetConfig, optional
            Configuration object. Creates default if None.
        dry_run : bool, default False
            If True, skip actual operations and only log.
        data_type : {'buildings', 'addresses', 'both'}, default 'both'
            Which dataset(s) to process.
        """
        super().__init__(config=config, dry_run=dry_run)
        self.data_type = data_type
        self.use_coiled = use_coiled
        self._coiled_cluster = None

    def get_coiled_cluster(self):
        """Get or create a Coiled cluster for distributed processing."""
        if self._coiled_cluster is None:
            console.log('Creating Coiled cluster for distributed processing...')

            cluster_config = {
                'name': f'ocr-{self.dataset_name}',
                'region': self.config.s3_region,
                'n_workers': [self.COILED_WORKERS, self.COILED_WORKERS, self.COILED_WORKERS],
                'tags': {'Project': 'OCR'},
                'worker_vm_types': [self.COILED_WORKER_VM],
                'scheduler_vm_types': [self.COILED_SCHEDULER_VM],
                'spot_policy': 'spot_with_fallback',
                'software': self.COILED_SOFTWARE,
                'wait_for_workers': 1,
                'idle_timeout': '5 minutes',
                'worker_disk_size': 150,
            }

            self._coiled_cluster = coiled.Cluster(**cluster_config)

            console.log(f'Coiled cluster created: {self._coiled_cluster.dashboard_link or "N/A"}')
        return self._coiled_cluster

    @property
    def s3_buildings_key(self) -> str:
        """S3 key for buildings parquet file."""
        return f'{self.config.base_prefix}/vector/{self.dataset_name}/CONUS-overture-buildings-{self.version}.parquet'

    @property
    def s3_addresses_key(self) -> str:
        """S3 key for addresses parquet file."""
        return f'{self.config.base_prefix}/vector/{self.dataset_name}/CONUS-overture-addresses-{self.version}.parquet'

    @staticmethod
    def _subset_buildings(
        version: str,
        bbox: tuple[float, float, float, float],
        overture_bucket: str,
        output_s3_uri: str,
        dry_run: bool = False,
    ) -> None:
        """Subset Overture buildings data to CONUS bbox."""
        import duckdb

        from ocr.console import console
        from ocr.utils import apply_s3_creds, install_load_extensions

        console.log(f'Subsetting Overture buildings (release {version}) to CONUS...')

        if dry_run:
            console.log(
                f'[DRY RUN] Would query buildings from s3://{overture_bucket}/release/{version}/theme=buildings/'
            )
            console.log(f'[DRY RUN] Would write to {output_s3_uri}')
            return

        install_load_extensions()
        apply_s3_creds()

        console.log(f'Querying buildings with bbox: {bbox}')
        console.log(f'Output: {output_s3_uri}')

        query = f"""
        COPY (
            SELECT bbox, geometry
            FROM read_parquet('s3://{overture_bucket}/release/{version}/theme=buildings/type=building/*.parquet')
            WHERE
                bbox.xmin BETWEEN {bbox[0]} AND {bbox[2]} AND
                bbox.ymin BETWEEN {bbox[1]} AND {bbox[3]}
        ) TO '{output_s3_uri}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """

        duckdb.sql(query)
        console.log(f'Successfully wrote buildings to {output_s3_uri}')

    @staticmethod
    def _subset_addresses(
        version: str,
        bbox: tuple[float, float, float, float],
        overture_bucket: str,
        output_s3_uri: str,
        dry_run: bool = False,
    ) -> None:
        """Subset Overture addresses data to CONUS bbox."""
        import duckdb

        from ocr.console import console
        from ocr.utils import apply_s3_creds, install_load_extensions

        console.log(f'Subsetting Overture addresses (release {version}) to CONUS...')

        if dry_run:
            console.log(
                f'[DRY RUN] Would query addresses from s3://{overture_bucket}/release/{version}/theme=addresses/'
            )
            console.log(f'[DRY RUN] Would write to {output_s3_uri}')
            return

        install_load_extensions()
        apply_s3_creds()

        console.log(f'Querying addresses with bbox: {bbox}')
        console.log(f'Output: {output_s3_uri}')

        query = f"""
        COPY (
            SELECT *
            FROM read_parquet('s3://{overture_bucket}/release/{version}/theme=addresses/type=address/*.parquet')
            WHERE
                bbox.xmin BETWEEN {bbox[0]} AND {bbox[2]} AND
                bbox.ymin BETWEEN {bbox[1]} AND {bbox[3]}
        ) TO '{output_s3_uri}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """

        duckdb.sql(query)
        console.log(f'Successfully wrote addresses to {output_s3_uri}')

    def download(self) -> None:
        """Download is not needed - data is queried directly from Overture S3."""
        console.log('Skipping download - Overture data is queried directly from S3')

    def process(self) -> None:
        """Process Overture data by subsetting to CONUS bbox and writing to S3."""
        console.log(f'Processing Overture Maps data (type: {self.data_type})...')
        client = None
        if self.use_coiled:
            console.log('Running download on Coiled cluster...')
            cluster = self.get_coiled_cluster()
            client = cluster.get_client()

        if self.data_type in ('buildings', 'both'):
            if client:
                future = client.submit(
                    self._subset_buildings,
                    version=self.version,
                    bbox=self.CONUS_BBOX,
                    overture_bucket=self.OVERTURE_BUCKET,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_buildings_key}',
                    dry_run=self.dry_run,
                )
                result = future.result()
                console.log(f'Subsetting buildings complete: {result}')
            else:
                self._subset_buildings(
                    version=self.version,
                    bbox=self.CONUS_BBOX,
                    overture_bucket=self.OVERTURE_BUCKET,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_buildings_key}',
                    dry_run=self.dry_run,
                )

        if self.data_type in ('addresses', 'both'):
            if client:
                future = client.submit(
                    self._subset_addresses,
                    version=self.version,
                    bbox=self.CONUS_BBOX,
                    overture_bucket=self.OVERTURE_BUCKET,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_addresses_key}',
                    dry_run=self.dry_run,
                )
                result = future.result()
                console.log(f'Subsetting addresses complete: {result}')
            else:
                self._subset_addresses(
                    version=self.version,
                    bbox=self.CONUS_BBOX,
                    overture_bucket=self.OVERTURE_BUCKET,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_addresses_key}',
                    dry_run=self.dry_run,
                )

        console.log('Overture processing complete')
