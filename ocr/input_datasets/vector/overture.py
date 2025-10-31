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
    COILED_WORKER_VM: ClassVar[str] = 'r8g.12xlarge'
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

    @property
    def s3_region_tagged_buildings_key(self) -> str:
        """S3 key for region-tagged buildings parquet file."""
        return f'{self.config.base_prefix}/vector/{self.dataset_name}/CONUS-overture-region-tagged-buildings-{self.version}.parquet'

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

    @staticmethod
    def _create_region_tagged_buildings(
        buildings_s3_uri: str,
        blocks_s3_uri: str,
        output_s3_uri: str,
        dry_run: bool = False,
    ) -> None:
        """Tag Overture buildings with census block information via spatial join.

        Performs a spatial join between Overture buildings and US Census blocks,
        adding geographic identifiers at multiple administrative levels (state,
        county, tract, block group, block).
        """
        import duckdb

        from ocr.console import console
        from ocr.utils import apply_s3_creds, install_load_extensions

        console.log('Creating region-tagged buildings dataset...')

        if dry_run:
            console.log(f'[DRY RUN] Would read buildings from {buildings_s3_uri}')
            console.log(f'[DRY RUN] Would read census blocks from {blocks_s3_uri}')
            console.log(f'[DRY RUN] Would write to {output_s3_uri}')
            return

        install_load_extensions()
        apply_s3_creds()

        console.log('Loading census FIPS lookup table...')
        duckdb.sql("""
            CREATE TEMP TABLE fips_lookup AS
                SELECT
                    column0 AS STATE,
                    lpad(column1, 2, '0') AS state_fips,
                    lpad(column2, 3, '0') AS county_fips,
                    column3 AS county
                FROM read_csv_auto(
                    'http://www2.census.gov/geo/docs/reference/codes/files/national_county.txt',
                    header=False
                );
        """)

        console.log('Creating spatial indexes...')
        duckdb.sql(f"""
            CREATE TEMP TABLE buildings AS
                SELECT bbox, geometry FROM read_parquet('{buildings_s3_uri}');
        """)
        duckdb.sql('CREATE INDEX buildings_idx ON buildings USING RTREE (geometry);')

        duckdb.sql(f"""
            CREATE TEMP TABLE blocks AS
                SELECT GEOID, bbox, geometry FROM read_parquet('{blocks_s3_uri}');
        """)
        duckdb.sql('CREATE INDEX blocks_idx ON blocks USING RTREE (geometry);')

        console.log('Performing spatial join...')
        query = f"""
        COPY (
            SELECT
                a.geometry,
                a.bbox,
                b.GEOID as block_geoid,
                SUBSTRING(b.GEOID, 1, 12) as block_group_geoid,
                SUBSTRING(b.GEOID, 1, 11) as tract_geoid,
                SUBSTRING(b.GEOID, 1, 5) as county_geoid,
                SUBSTRING(b.GEOID, 1, 2) as state_fips,
                SUBSTRING(b.GEOID, 3, 3) as county_fips,
                SUBSTRING(b.GEOID, 5, 6) as tract_fips,
                SUBSTRING(b.GEOID, 11, 1) as block_group_fips,
                SUBSTRING(b.GEOID, 12, 4) as block_fips,
                c.STATE as state_abbrev,
                c.county as county_name
            FROM buildings a
            JOIN blocks b
                ON a.bbox.xmin <= b.bbox.xmax
                AND a.bbox.xmax >= b.bbox.xmin
                AND a.bbox.ymin <= b.bbox.ymax
                AND a.bbox.ymax >= b.bbox.ymin
                AND ST_Intersects(a.geometry, b.geometry)
            LEFT JOIN fips_lookup c
                ON SUBSTRING(b.GEOID, 1, 2) = c.state_fips
                AND SUBSTRING(b.GEOID, 3, 3) = c.county_fips
        ) TO '{output_s3_uri}' (FORMAT 'parquet', OVERWRITE_OR_IGNORE true, COMPRESSION 'zstd');
        """

        duckdb.sql(query)
        console.log(f'Successfully wrote region-tagged buildings to {output_s3_uri}')

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

        # Create region-tagged buildings if we processed buildings
        if self.data_type in ('buildings', 'both'):
            console.log('Creating region-tagged buildings dataset...')

            from ocr import catalog

            buildings_s3_uri = f's3://{self.config.s3_bucket}/{self.s3_buildings_key}'

            blocks_dataset = catalog.get_dataset('us-census-blocks')
            blocks_s3_uri = f's3://{blocks_dataset.bucket}/{blocks_dataset.prefix}'

            output_s3_uri = f's3://{self.config.s3_bucket}/{self.s3_region_tagged_buildings_key}'

            if client:
                future = client.submit(
                    self._create_region_tagged_buildings,
                    buildings_s3_uri=buildings_s3_uri,
                    blocks_s3_uri=blocks_s3_uri,
                    output_s3_uri=output_s3_uri,
                    dry_run=self.dry_run,
                )
                result = future.result()
                console.log(f'Region tagging complete: {result}')
            else:
                self._create_region_tagged_buildings(
                    buildings_s3_uri=buildings_s3_uri,
                    blocks_s3_uri=blocks_s3_uri,
                    output_s3_uri=output_s3_uri,
                    dry_run=self.dry_run,
                )

        console.log('Overture processing complete')
