"""US Census TIGER/Line geographic boundary datasets.

This module handles downloading and processing Census TIGER/Line shapefiles:
1. Download TIGER/Line shapefiles (blocks, tracts, counties) from Census Bureau
2. Convert to GeoParquet format with spatial metadata
3. Upload to S3

Reference:
https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
"""

from typing import ClassVar, Literal

import coiled
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from ocr.console import console
from ocr.input_datasets.base import BaseDatasetProcessor, InputDatasetConfig

# FIPS codes for CONUS states + DC (excludes Alaska and Hawaii)
FIPS_CODES: dict[str, str] = {
    'Alabama': '01',
    'Arizona': '04',
    'Arkansas': '05',
    'California': '06',
    'Colorado': '08',
    'Connecticut': '09',
    'Delaware': '10',
    'District of Columbia': '11',
    'Florida': '12',
    'Georgia': '13',
    'Idaho': '16',
    'Illinois': '17',
    'Indiana': '18',
    'Iowa': '19',
    'Kansas': '20',
    'Kentucky': '21',
    'Louisiana': '22',
    'Maine': '23',
    'Maryland': '24',
    'Massachusetts': '25',
    'Michigan': '26',
    'Minnesota': '27',
    'Mississippi': '28',
    'Missouri': '29',
    'Montana': '30',
    'Nebraska': '31',
    'Nevada': '32',
    'New Hampshire': '33',
    'New Jersey': '34',
    'New Mexico': '35',
    'New York': '36',
    'North Carolina': '37',
    'North Dakota': '38',
    'Ohio': '39',
    'Oklahoma': '40',
    'Oregon': '41',
    'Pennsylvania': '42',
    'Rhode Island': '44',
    'South Carolina': '45',
    'South Dakota': '46',
    'Tennessee': '47',
    'Texas': '48',
    'Utah': '49',
    'Vermont': '50',
    'Virginia': '51',
    'Washington': '53',
    'West Virginia': '54',
    'Wisconsin': '55',
    'Wyoming': '56',
}


class CensusTigerProcessor(BaseDatasetProcessor):
    """Processor for US Census TIGER/Line geographic boundaries."""

    dataset_name: str = 'census-tiger'
    dataset_type = 'vector'
    description: str = (
        'US Census TIGER/Line geographic boundaries (blocks, tracts, counties, states, nation)'
    )
    source_url: str = (
        'https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html'
    )
    version: str = '2024'  # TIGER vintage year for tracts/counties/states/nation
    blocks_version: str = '2025'  # Blocks use 2025 vintage

    # Coiled configuration
    COILED_WORKER_VM: ClassVar[str] = 'c8g.8xlarge'
    COILED_SCHEDULER_VM: ClassVar[str] = 'm8g.xlarge'
    COILED_WORKERS: ClassVar[int] = 1
    COILED_SOFTWARE: ClassVar[str | None] = None

    def __init__(
        self,
        config: InputDatasetConfig | None = None,
        *,
        dry_run: bool = False,
        geography_type: Literal['blocks', 'tracts', 'counties', 'states', 'nation', 'all'] = 'all',
        subset_states: list[str] | None = None,
        use_coiled: bool = False,
        coiled_software: str | None = None,
    ):
        """Initialize the Census TIGER processor.

        Parameters
        ----------
        config : InputDatasetConfig, optional
            Configuration object. Creates default if None.
        dry_run : bool, default False
            If True, skip actual operations and only log.
        geography_type : {'blocks', 'tracts', 'counties', 'states', 'nation', 'all'}, default 'all'
            Which geography type(s) to process.
        subset_states : list[str], optional
            List of state names to process (e.g., ['California', 'Oregon']).
            If None, processes all CONUS states + DC.
        use_coiled : bool, default False
            Whether to use Coiled for distributed processing.
        coiled_software : str, optional
            Name of Coiled software environment to use.
        """
        super().__init__(config=config, dry_run=dry_run)
        self.geography_type = geography_type
        self.subset_states = subset_states
        self.use_coiled = use_coiled
        self._coiled_cluster = None

        # Override class-level COILED_SOFTWARE if provided
        if coiled_software is not None:
            self.COILED_SOFTWARE = coiled_software

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
    def s3_blocks_key(self) -> str:
        """S3 key for blocks parquet file."""
        return f'{self.config.base_prefix}/vector/{self.dataset_name}/blocks/blocks.parquet'

    @property
    def s3_tracts_key(self) -> str:
        """S3 base path for tracts parquet file."""
        return f'{self.config.base_prefix}/vector/{self.dataset_name}/tracts/tracts.parquet'

    @property
    def s3_counties_key(self) -> str:
        """S3 key for counties parquet file."""
        return f'{self.config.base_prefix}/vector/{self.dataset_name}/counties/counties.parquet'

    @property
    def s3_states_key(self) -> str:
        """S3 key for states parquet file."""
        return f'{self.config.base_prefix}/vector/{self.dataset_name}/states/states.parquet'

    @property
    def s3_nation_key(self) -> str:
        """S3 key for nation parquet file."""
        return f'{self.config.base_prefix}/vector/{self.dataset_name}/nation/nation.parquet'

    def _get_fips_codes(self) -> dict[str, str]:
        """Get FIPS codes for selected states."""
        if self.subset_states:
            return {state: FIPS_CODES[state] for state in self.subset_states if state in FIPS_CODES}
        return FIPS_CODES

    @staticmethod
    def _process_blocks(
        fips_codes: dict[str, str],
        blocks_version: str,
        output_s3_uri: str,
        dry_run: bool = False,
    ) -> None:
        """Process Census blocks for all states into a single GeoParquet file."""
        from ocr.console import console

        console.log(f'Processing Census blocks (TIGER {blocks_version})...')

        if dry_run:
            console.log(f'[DRY RUN] Would download blocks for {len(fips_codes)} states')
            console.log(f'[DRY RUN] Would write to {output_s3_uri}')
            return

        gdfs = []
        for state, fips in tqdm(fips_codes.items(), desc='Processing blocks'):
            block_url = f'https://www2.census.gov/geo/tiger/TIGER{blocks_version}/TABBLOCKSUFX/tl_{blocks_version}_{fips}_tabblocksufx.zip'
            console.log(f'Reading {state} blocks from {block_url}')
            gdf = gpd.read_file(block_url)
            gdfs.append(gdf)

        console.log('Combining all blocks into single GeoDataFrame...')
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

        console.log(f'Writing {len(combined_gdf)} blocks to {output_s3_uri}')
        combined_gdf.to_parquet(
            output_s3_uri,
            compression='zstd',
            geometry_encoding='WKB',
            write_covering_bbox=True,
            schema_version='1.1.0',
        )
        console.log(f'Successfully wrote blocks to {output_s3_uri}')

    @staticmethod
    def _process_tracts(
        fips_codes: dict[str, str],
        tracts_version: str,
        output_s3_uri: str,
        dry_run: bool = False,
    ) -> None:
        """Process Census tracts for all states into a single GeoParquet file."""

        from ocr.console import console

        console.log(f'Processing Census tracts (TIGER {tracts_version})...')

        if dry_run:
            console.log(f'[DRY RUN] Would download tracts for {len(fips_codes)} states')
            console.log(f'[DRY RUN] Would write to {output_s3_uri}')
            return

        gdfs = []
        for state, fips in tqdm(fips_codes.items(), desc='Processing tracts'):
            tract_url = f'https://www2.census.gov/geo/tiger/TIGER{tracts_version}/TRACT/tl_{tracts_version}_{fips}_tract.zip'
            console.log(f'Reading {state} tracts from {tract_url}')
            gdf = gpd.read_file(tract_url)
            gdfs.append(gdf)

        console.log('Combining all tracts into single GeoDataFrame.')
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

        console.log(f'Writing {len(combined_gdf)} tracts to {output_s3_uri}')
        combined_gdf.to_parquet(
            output_s3_uri,
            compression='zstd',
            geometry_encoding='WKB',
            write_covering_bbox=True,
            schema_version='1.1.0',
        )
        console.log(f'Successfully wrote tracts to {output_s3_uri}')

    @staticmethod
    def _process_counties(
        counties_version: str,
        output_s3_uri: str,
        dry_run: bool = False,
    ) -> None:
        """Process US counties into a single GeoParquet file."""

        from ocr.console import console

        console.log(f'Processing US counties (TIGER {counties_version})...')

        if dry_run:
            console.log('[DRY RUN] Would download national counties shapefile')
            console.log(f'[DRY RUN] Would write to {output_s3_uri}')
            return

        county_url = f'https://www2.census.gov/geo/tiger/TIGER{counties_version}/COUNTY/tl_{counties_version}_us_county.zip'

        console.log(f'Reading counties from {county_url}')
        gdf = gpd.read_file(county_url)

        console.log(f'Writing {len(gdf)} counties to {output_s3_uri}')
        gdf.to_parquet(
            output_s3_uri,
            compression='zstd',
            geometry_encoding='WKB',
            write_covering_bbox=True,
            schema_version='1.1.0',
        )
        console.log(f'Successfully wrote counties to {output_s3_uri}')

    @staticmethod
    def _process_states(
        states_version: str,
        output_s3_uri: str,
        dry_run: bool = False,
    ) -> None:
        """Process US states into a single GeoParquet file"""

        from ocr.console import console

        if dry_run:
            console.log(f'[DRY RUN] Would write to {output_s3_uri}')
            return

        # Using 500k scale cartographic boundary file
        states_url = f'https://www2.census.gov/geo/tiger/GENZ{states_version}/shp/cb_{states_version}_us_state_500k.zip'

        console.log(f'Reading states from {states_url}')
        gdf = gpd.read_file(states_url)

        # Filter to CONUS only using existing FIPS_CODES dictionary
        conus_fips = list(FIPS_CODES.values())
        gdf = gdf[gdf['STATEFP'].isin(conus_fips)]

        # Keep only essential columns
        gdf = gdf[['GEOID', 'STUSPS', 'NAME', 'geometry']]

        # Zero-pad GEOID to 2 digits (state FIPS codes)
        gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(2)

        console.log(f'Writing {len(gdf)} CONUS states to {output_s3_uri}')
        gdf.to_parquet(
            output_s3_uri,
            compression='zstd',
            geometry_encoding='WKB',
            write_covering_bbox=True,
            schema_version='1.1.0',
        )
        console.log(f'Successfully wrote states to {output_s3_uri}')

    @staticmethod
    def _process_nation(
        nation_version: str,
        output_s3_uri: str,
        dry_run: bool = False,
    ) -> None:
        """Process CONUS into a single GeoParquet file"""

        from ocr.console import console

        if dry_run:
            console.log(f'[DRY RUN] Would write to {output_s3_uri}')
            return

        nation_url = f'https://www2.census.gov/geo/tiger/GENZ{nation_version}/shp/cb_{nation_version}_us_nation_5m.zip'

        console.log(f'Reading nation outline from {nation_url}')
        gdf = gpd.read_file(nation_url)

        conus_bbox = (-125.0, 24.0, -66.0, 50.0)

        console.log(f'Clipping to CONUS bounding box: {conus_bbox}')
        from shapely.geometry import box

        conus_box = box(*conus_bbox)
        gdf = gdf.clip(conus_box)

        gdf = gdf[['GEOID', 'NAME', 'geometry']]

        console.log(f'Writing CONUS nation outline to {output_s3_uri}')
        gdf.to_parquet(
            output_s3_uri,
            compression='zstd',
            geometry_encoding='WKB',
            write_covering_bbox=True,
            schema_version='1.1.0',
        )
        console.log(f'wrote nation outline to: {output_s3_uri}')

    def download(self) -> None:
        """Download is not needed - data is downloaded directly during processing."""
        console.log('Skipping download - Census TIGER data is downloaded during processing')

    def process(self) -> None:
        """Process Census TIGER data by downloading and converting to GeoParquet."""
        client = None
        console.log(f'Processing Census TIGER data (type: {self.geography_type})...')
        if self.use_coiled:
            console.log('Running download on Coiled cluster...')
            cluster = self.get_coiled_cluster()
            client = cluster.get_client()

        fips_codes = self._get_fips_codes()
        console.log(f'Processing {len(fips_codes)} states/territories')

        if self.geography_type in ('blocks', 'all'):
            if client:
                console.log('Submitting block processing to Coiled cluster...')
                future = client.submit(
                    self._process_blocks,
                    fips_codes=fips_codes,
                    blocks_version=self.blocks_version,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_blocks_key}',
                    dry_run=self.dry_run,
                )
                future.result()
            else:
                self._process_blocks(
                    fips_codes=fips_codes,
                    blocks_version=self.blocks_version,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_blocks_key}',
                    dry_run=self.dry_run,
                )

        if self.geography_type in ('tracts', 'all'):
            if client:
                console.log('Submitting tracts processing to Coiled cluster...')
                future = client.submit(
                    self._process_tracts,
                    fips_codes=fips_codes,
                    tracts_version=self.version,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_tracts_key}',
                    dry_run=self.dry_run,
                )
                future.result()
            else:
                self._process_tracts(
                    fips_codes=fips_codes,
                    tracts_version=self.version,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_tracts_key}',
                    dry_run=self.dry_run,
                )

        if self.geography_type in ('counties', 'all'):
            if client:
                console.log('Submitting county processing to Coiled cluster...')
                future = client.submit(
                    self._process_counties,
                    counties_version=self.version,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_counties_key}',
                    dry_run=self.dry_run,
                )
                future.result()
            else:
                self._process_counties(
                    counties_version=self.version,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_counties_key}',
                    dry_run=self.dry_run,
                )

        if self.geography_type in ('states', 'all'):
            if client:
                console.log('Submitting states processing to Coiled cluster...')
                future = client.submit(
                    self._process_states,
                    states_version=self.version,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_states_key}',
                    dry_run=self.dry_run,
                )
                future.result()
            else:
                self._process_states(
                    states_version=self.version,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_states_key}',
                    dry_run=self.dry_run,
                )

        if self.geography_type in ('nation', 'all'):
            if client:
                console.log('Submitting nation processing to Coiled cluster...')
                future = client.submit(
                    self._process_nation,
                    nation_version=self.version,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_nation_key}',
                    dry_run=self.dry_run,
                )
                future.result()
            else:
                self._process_nation(
                    nation_version=self.version,
                    output_s3_uri=f's3://{self.config.s3_bucket}/{self.s3_nation_key}',
                    dry_run=self.dry_run,
                )

        console.log('Census TIGER processing complete')
