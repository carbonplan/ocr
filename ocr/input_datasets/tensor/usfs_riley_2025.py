"""USFS Spatial datasets of probabilistic wildfire risk components for the conterminous United States
for circa 2011 climate and projected future climate circa 2047 (Riley et al., 2025, RDS-2025-0006)

This module handles the processing pipeline for the USFS Riley et al., 2025 dataset:
1. Download ZIP archive containing 2011 and 2047 climate runs from USFS Box
2. Extract and upload TIFFs to S3
3. Merge TIFFs for each climate run into separate Icechunk stores (EPSG:5070)
4. Reproject from EPSG:5070 to EPSG:4326 and interpolate from 270m to 30m

Reference:
https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0006
"""

import tempfile
from pathlib import Path
from typing import ClassVar

import coiled
import dask.base
import pooch
import xarray as xr
from odc.geo.xr import assign_crs, xr_reproject

from ocr import catalog
from ocr.console import console
from ocr.input_datasets.base import BaseDatasetProcessor, InputDatasetConfig
from ocr.input_datasets.storage import IcechunkReader, IcechunkWriter, S3Uploader


class RileyEtAl2025Processor(BaseDatasetProcessor):
    """Processor for USFS Riley et al., 2025 wildfire risk dataset (RDS-2025-0006).

    This dataset contains probabilistic wildfire risk components for two climate scenarios:
    - 2011ClimateRun: circa 2011 climate (270m resolution)
    - 2047ClimateRun: projected future climate circa 2047 (270m resolution)

    Each climate run contains 7 variables: BP, FLP1, FLP2, FLP3, FLP4, FLP5, FLP6
    """

    dataset_name: str = 'riley-et-al-2025'
    dataset_type = 'tensor'
    description: str = (
        'Spatial datasets of probabilistic wildfire risk components for the conterminous '
        'United States (270m) for circa 2011 climate and projected future climate circa 2047'
    )
    source_url: str = 'https://www.fs.usda.gov/rds/archive/catalog/RDS-2025-0006'
    version: str = '2025'
    rds_id: str = 'RDS-2025-0006'

    coiled_software: str | None = None
    # Processing configuration (dataset-specific tuning)
    COILED_WORKERS: ClassVar[int] = 2
    COILED_WORKER_VM: ClassVar[str] = 'r7a.24xlarge'
    COILED_SCHEDULER_VM: ClassVar[str] = 'r7a.xlarge'
    CHUNK_SIZES: ClassVar[dict[str, int]] = {'y': 6000, 'x': 4500}
    REPROJECTED_CHUNK_SIZES: ClassVar[dict[str, int]] = {'latitude': 6000, 'longitude': 4500}

    # Climate runs and their variables
    CLIMATE_RUNS: ClassVar[list[str]] = ['2011ClimateRun', '2047ClimateRun']
    VARIABLES: ClassVar[list[str]] = ['BP', 'FLP1', 'FLP2', 'FLP3', 'FLP4', 'FLP5', 'FLP6']

    # Source archive
    ARCHIVE_URL: ClassVar[str] = (
        'https://usfs-public.box.com/shared/static/h55qel755s97nagdu97ebd4z6fzpp3w1.zip'
    )
    ARCHIVE_HASH: ClassVar[str | None] = (
        'a92b4f6ecb2c5f3c496851a82c5b65d38b7c97cfd6c16d433cafaa5905265825'
    )

    def __init__(
        self,
        config: InputDatasetConfig | None = None,
        *,
        dry_run: bool = False,
        use_coiled: bool = False,
    ):
        """Initialize the Riley et al. 2025 processor."""
        super().__init__(dry_run=dry_run, config=config)
        self.use_coiled = use_coiled
        self._coiled_cluster = None

    @property
    def s3_tiff_prefix(self) -> str:
        """S3 prefix for storing extracted TIFFs."""
        return f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/raw-input-tiffs'

    @property
    def s3_icechunk_prefix_2011(self) -> str:
        """S3 prefix for 2011 climate run Icechunk store (EPSG:5070)."""
        return f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/2011-climate-run-270m-5070.icechunk'

    @property
    def s3_icechunk_prefix_2047(self) -> str:
        """S3 prefix for 2047 climate run Icechunk store (EPSG:5070)."""
        return f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/2047-climate-run-270m-5070.icechunk'

    @property
    def s3_icechunk_4326_prefix_2011(self) -> str:
        """S3 prefix for reprojected 2011 climate run (EPSG:4326, 30m)."""
        return f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/2011-climate-run-30m-4326.icechunk'

    @property
    def s3_icechunk_4326_prefix_2047(self) -> str:
        """S3 prefix for reprojected 2047 climate run (EPSG:4326, 30m)."""
        return f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/2047-climate-run-30m-4326.icechunk'

    def get_coiled_cluster(self):
        """Get or create a Coiled cluster for distributed processing."""
        if self._coiled_cluster is None:
            console.log('Creating Coiled cluster for distributed processing...')

            cluster_config = {
                'name': f'ocr-{self.dataset_name}',
                'region': self.config.s3_region,
                'n_workers': [self.COILED_WORKERS, self.COILED_WORKERS, self.COILED_WORKERS * 10],
                'tags': {'Project': 'OCR'},
                'worker_vm_types': [self.COILED_WORKER_VM],
                'scheduler_vm_types': [self.COILED_SCHEDULER_VM],
                'spot_policy': 'spot_with_fallback',
                'software': self.coiled_software,
                'wait_for_workers': 1,
                'idle_timeout': '5 minutes',
                'worker_disk_size': 150,
            }

            self._coiled_cluster = coiled.Cluster(**cluster_config)

            console.log(f'Coiled cluster created: {self._coiled_cluster.dashboard_link or "N/A"}')
        return self._coiled_cluster

    @staticmethod
    def _download_and_upload_archive(
        archive_url: str,
        archive_hash: str | None,
        rds_id: str,
        s3_tiff_prefix: str,
        s3_bucket: str,
        s3_region: str,
    ) -> dict[str, list[str]]:
        """Download archive and upload all TIFFs to S3.

        This is a static method so it can be serialized and executed on Coiled workers.

        Returns
        -------
        dict[str, list[str]]
            Mapping of climate run names to lists of S3 URIs
        """

        # Download and extract using pooch
        cache_path = Path(tempfile.gettempdir())
        cache_path.mkdir(parents=True, exist_ok=True)

        extracted_files = pooch.retrieve(
            url=archive_url,
            known_hash=archive_hash,
            fname=f'{rds_id}.zip',
            path=cache_path,
            processor=pooch.Unzip(),
            progressbar=True,
        )

        # Find all TIFF files organized by climate run
        if isinstance(extracted_files, str):
            extracted_files = [extracted_files]

        tiff_files = [Path(f) for f in extracted_files if f.endswith('.tif')]
        if not tiff_files:
            raise FileNotFoundError('No .tif files found in extracted archive')

        # Upload TIFFs to S3, preserving directory structure
        uploader = S3Uploader(s3_bucket, s3_region, dry_run=False)
        s3_uris = {}

        for tiff_path in tiff_files:
            # Extract climate run from path: USFS_fire_risk/Data/2011ClimateRun/BP.tif
            parts = tiff_path.parts
            if 'Data' in parts:
                data_idx = parts.index('Data')
                climate_run = parts[data_idx + 1]
                var_name = tiff_path.stem

                s3_key = f'{s3_tiff_prefix}/Data/{climate_run}/{var_name}.tif'
                s3_uri = uploader.upload_file(tiff_path, s3_key)

                if climate_run not in s3_uris:
                    s3_uris[climate_run] = []
                s3_uris[climate_run].append(s3_uri)

        return s3_uris

    def download(self) -> None:
        """Download archive from USFS Box, extract, and upload TIFFs to S3.

        If use_coiled=True, runs download on Coiled worker.
        """
        console.log(f'Downloading {self.rds_id} archive from USFS Box...')

        if self.use_coiled:
            console.log('Running download on Coiled cluster...')
            cluster = self.get_coiled_cluster()
            client = cluster.get_client()

            future = client.submit(
                self._download_and_upload_archive,
                archive_url=self.ARCHIVE_URL,
                archive_hash=self.ARCHIVE_HASH,
                rds_id=self.rds_id,
                s3_tiff_prefix=self.s3_tiff_prefix,
                s3_bucket=self.config.s3_bucket,
                s3_region=self.config.s3_region,
            )

            try:
                s3_uris = future.result()
                for climate_run, uris in s3_uris.items():
                    console.log(f'✓ {climate_run}: {len(uris)} files uploaded')
            except Exception as e:
                console.log(f'[bold red]✗ Download failed: {e}[/bold red]')
                raise

            console.log('Download and upload completed on Coiled')
        else:
            # Local execution
            s3_uris = self._download_and_upload_archive(
                archive_url=self.ARCHIVE_URL,
                archive_hash=self.ARCHIVE_HASH,
                rds_id=self.rds_id,
                s3_tiff_prefix=self.s3_tiff_prefix,
                s3_bucket=self.config.s3_bucket,
                s3_region=self.config.s3_region,
            )

            console.log('Download and upload completed locally')

    def merge_tiffs_to_icechunk(self, climate_run: str) -> None:
        """Merge all TIFF variables for a climate run into a single Icechunk store.

        Parameters
        ----------
        climate_run : str
            Climate run name ('2011ClimateRun' or '2047ClimateRun')
        """
        console.log(f'Merging TIFFs for {climate_run} into Icechunk store...')

        # Build list of S3 URLs for all variables
        s3_path = f's3://{self.config.s3_bucket}/{self.s3_tiff_prefix}/Data/{climate_run}/'
        url_list = [f'{s3_path}{var}.tif' for var in self.VARIABLES]

        # Preprocess function to rename band_data to variable name
        def preprocess(ds, filename):
            return ds.rename({'band_data': filename})

        # Open and combine all TIFFs
        console.log(f'Loading {len(url_list)} TIFFs from S3...')
        combined_ds = xr.open_mfdataset(
            url_list,
            combine='by_coords',
            preprocess=lambda ds: preprocess(
                ds, ds.encoding['source'].split('/')[-1].split('.')[0]
            ),
            engine='rasterio',
            parallel=True,
        )

        # Remove 1D band dimension and chunk
        combined_ds = combined_ds.squeeze().drop_vars('band').chunk(self.CHUNK_SIZES)

        # Write to Icechunk
        icechunk_prefix = (
            self.s3_icechunk_prefix_2011 if '2011' in climate_run else self.s3_icechunk_prefix_2047
        )

        writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=icechunk_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )
        writer.write(combined_ds, 'Add all raster data to store')

        console.log(f'Successfully merged {climate_run} TIFFs to Icechunk store')

    def reproject_to_4326(self, climate_run_year: str) -> None:
        """Reproject climate run from EPSG:5070 to EPSG:4326 and interpolate to 30m.

        Parameters
        ----------
        climate_run_year : str
            Climate run year ('2011' or '2047')
        """
        console.log(
            f'Reprojecting {climate_run_year} climate run to EPSG:4326 and interpolating to 30m...'
        )

        if self.use_coiled:
            client = self.get_coiled_cluster().get_client()
            console.log(f'Using Coiled cluster: {client}')

        # Load the EPSG:5070 dataset
        icechunk_prefix = (
            self.s3_icechunk_prefix_2011
            if climate_run_year == '2011'
            else self.s3_icechunk_prefix_2047
        )

        reader = IcechunkReader(
            bucket=self.config.s3_bucket,
            prefix=icechunk_prefix,
            region=self.config.s3_region,
            debug=self.config.debug,
        )

        session = reader.repo.readonly_session('main')
        console.log('Loading dataset from Icechunk store...')
        climate_ds = xr.open_zarr(session.store).persist()

        # Downcast to float32
        for var in list(climate_ds):
            climate_ds[var] = climate_ds[var].astype('float32')

        console.log(f'Pre-reproject: {climate_ds}')

        # Assign CRS and load target dataset for interpolation
        climate_ds = assign_crs(climate_ds, crs='EPSG:5070')

        rps_30_4326 = catalog.get_dataset('scott-et-al-2024-30m-4326').to_xarray()
        rps_30_4326 = assign_crs(rps_30_4326, 'EPSG:4326')

        # Reproject to match target geobox
        console.log('Reprojecting to EPSG:4326...')
        climate_4326 = xr_reproject(climate_ds, how=rps_30_4326.odc.geobox)

        # Clip negative burn probabilities to 0
        climate_4326 = climate_4326.clip(min=0)
        climate_4326 = climate_4326.sortby(['latitude', 'longitude']).chunk(
            self.REPROJECTED_CHUNK_SIZES
        )
        climate_4326 = dask.base.optimize(climate_4326)[0]

        # force the canonical coordinates from scott-et-al-2024 onto reprojected riley dataset
        climate_4326 = climate_4326.assign_coords(
            longitude=rps_30_4326.longitude, latitude=rps_30_4326.latitude
        )

        console.log(f'Post-reproject: {climate_4326}')

        # Assign processing attributes
        climate_4326.attrs = {
            'title': f'{self.rds_id}',
            'version': self.version,
            'data_source': self.source_url,
            'description': (
                f'Modified version of: {self.description}. '
                'This dataset was created by combining multiple source TIFF files, '
                'interpolating from 270m to 30m and projecting from EPSG:5070 to EPSG:4326 using USFS Scott et al. (2024) as reference. '
                'It is stored in the Icechunk storage format.'
            ),
            'EPSG': '4326',
            'resolution': '30m',
            'climate_run_year': climate_run_year,
        }

        # Write to Icechunk
        icechunk_4326_prefix = (
            self.s3_icechunk_4326_prefix_2011
            if climate_run_year == '2011'
            else self.s3_icechunk_4326_prefix_2047
        )

        writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=icechunk_4326_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )
        writer.write(
            climate_4326, f'Reproject {climate_run_year} to EPSG:4326 and interpolate to 30m'
        )

        console.log(f'Successfully reprojected {climate_run_year} climate run to EPSG:4326')

    def process(self) -> None:
        """Run the complete processing pipeline for both climate runs."""

        if self.use_coiled:
            client = self.get_coiled_cluster().get_client()
            console.log(f'Using Coiled cluster: {client}')

        # Merge TIFFs for both climate runs
        for climate_run in self.CLIMATE_RUNS:
            self.merge_tiffs_to_icechunk(climate_run)

        # Reproject both climate runs
        for year in ['2011', '2047']:
            self.reproject_to_4326(year)

        # Clean up Coiled cluster if created
        if self._coiled_cluster is not None:
            console.log('Shutting down Coiled cluster...')
            self._coiled_cluster.shutdown()
