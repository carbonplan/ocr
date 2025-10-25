"""USFS Wildfire Risk to Communities dataset (Scott et al. 2024, RDS-2020-0016-02).

This module handles the processing pipeline for the second edition of the USFS
wildfire risk dataset:
1. Download 8 variable TIFFs from USFS Box
2. Merge TIFFs into a single Icechunk store (EPSG:5070)
3. Reproject from EPSG:5070 to EPSG:4326

Reference:
https://www.fs.usda.gov/rds/archive/catalog/RDS-2020-0016-2
"""

import tempfile
from pathlib import Path
from typing import ClassVar

import coiled
import dask.base
import pooch
import xarray as xr
from odc.geo.xr import assign_crs, xr_reproject

from ocr.console import console
from ocr.input_datasets.base import BaseDatasetProcessor, InputDatasetConfig
from ocr.input_datasets.storage import IcechunkReader, IcechunkWriter, S3Uploader
from ocr.utils import get_temp_dir


class ScottEtAl2024Processor(BaseDatasetProcessor):
    """Processor for USFS Scott et al. 2024 wildfire risk dataset (RDS-2020-0016-02)."""

    # Dataset metadata
    dataset_name: str = 'scott-et-al-2024'
    dataset_type = 'tensor'
    description: str = (
        'Wildfire Risk to Communities: Spatial datasets of landscape-wide '
        'wildfire risk components for the United States (2nd Edition)'
    )
    source_url: str = 'https://www.fs.usda.gov/rds/archive/catalog/RDS-2020-0016-2'
    version: str = '2024-V2'
    rds_id: str = 'RDS-2020-0016-02'
    coiled_software: str | None = None
    # Processing configuration (dataset-specific tuning)
    COILED_WORKERS: ClassVar[int] = 10
    COILED_WORKER_VM: ClassVar[str] = 'm8g.large'
    COILED_SCHEDULER_VM: ClassVar[str] = 'm8g.2xlarge'
    CHUNK_SIZES: ClassVar[dict[str, int]] = {'y': 6000, 'x': 4500}
    REPROJECTED_CHUNK_SIZES: ClassVar[dict[str, int]] = {'latitude': 6000, 'longitude': 4500}

    # Variable definitions with URLs and optional hashes
    # TODO: Add known_hash values for each file for verification
    VARIABLES: ClassVar[dict[str, dict[str, str | None]]] = {
        'BP': {
            'url': 'https://usfs-public.box.com/shared/static/7itw7p56vje2m0u3kqh91lt6kqq1i9l1.zip',
            'hash': None,  # Add md5:... or sha256:... hash after first download
        },
        'CRPS': {
            'url': 'https://usfs-public.box.com/shared/static/v1wjt4r3pp0bjb05w8qfnlbm4y3m66q5.zip',
            'hash': None,
        },
        'CFL': {
            'url': 'https://usfs-public.box.com/shared/static/7nb6hpw2rfc0zrhk1mv80fhbirajoqfd.zip',
            'hash': None,
        },
        'Exposure': {
            'url': 'https://usfs-public.box.com/shared/static/nbmlha1iejzzjo9y3uoehln493o2c4ad.zip',
            'hash': None,
        },
        'FLEP4': {
            'url': 'https://usfs-public.box.com/shared/static/502cm6vef6axhtljvy6b8p35tqugg2ds.zip',
            'hash': None,
        },
        'FLEP8': {
            'url': 'https://usfs-public.box.com/shared/static/gwasv734wwcx77zc4wj4ntfhaxyt8mel.zip',
            'hash': None,
        },
        'RPS': {
            'url': 'https://usfs-public.box.com/shared/static/88tv8byot0t22o9p1eqlrfqco3z5ouvf.zip',
            'hash': None,
        },
        'WHP': {
            'url': 'https://usfs-public.box.com/shared/static/jz74xh0eqdezblhexwu2s2at7fqgom8n.zip',
            'hash': None,
        },
    }

    def __init__(
        self,
        config: InputDatasetConfig | None = None,
        *,
        dry_run: bool = False,
        use_coiled: bool = True,
    ):
        """Initialize the Scott et al. 2024 processor.

        Parameters
        ----------
        config : InputDatasetConfig, optional
            Configuration object. Creates default if None.
        dry_run : bool, default False
            If True, skip actual S3 writes and only log operations.
        use_coiled : bool, default False
            If True, use Coiled for distributed processing (merge and reproject steps).
        """
        super().__init__(config, dry_run=dry_run)
        self.use_coiled = use_coiled
        self._coiled_cluster = None

    @property
    def s3_tiff_prefix(self) -> str:
        """S3 prefix for raw TIFF files."""
        return f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/raw-input-tiffs'

    @property
    def s3_icechunk_prefix(self) -> str:
        """S3 prefix for merged Icechunk store (EPSG:5070)."""
        return f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/processed.icechunk'

    @property
    def s3_icechunk_4326_prefix(self) -> str:
        """S3 prefix for reprojected Icechunk store (EPSG:4326)."""
        return (
            f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/processed-30m-4326.icechunk'
        )

    def get_coiled_cluster(self):
        """Get or create a Coiled cluster for distributed processing."""
        if self._coiled_cluster is None:
            console.log('Creating Coiled cluster for distributed processing...')
            self._coiled_cluster = coiled.Cluster(
                name=f'ocr-{self.dataset_name}',
                region=self.config.s3_region,
                n_workers=[self.COILED_WORKERS, self.COILED_WORKERS, self.COILED_WORKERS * 10],
                tags={'Project': 'OCR'},
                worker_vm_types=[self.COILED_WORKER_VM],
                scheduler_vm_types=[self.COILED_SCHEDULER_VM],
                software=self.coiled_software,
                idle_timeout='10 minutes',
                spot_policy='spot_with_fallback',
                worker_disk_size=150,
            )
            console.log(f'Coiled cluster created: {self._coiled_cluster.dashboard_link or "N/A"}')
        return self._coiled_cluster

    @staticmethod
    def _download_and_upload_variable(
        var_name: str,
        var_info: dict[str, str | None],
        rds_id: str,
        s3_tiff_prefix: str,
        s3_bucket: str,
        s3_region: str,
    ) -> str:
        """Download a single variable and upload to S3.

        This is a static method so it can be serialized and executed on Coiled workers.

        Parameters
        ----------
        var_name : str
            Variable name (e.g., 'BP', 'CRPS')
        var_info : dict
            Dictionary with 'url' and 'hash' keys
        rds_id : str
            Dataset RDS ID
        s3_tiff_prefix : str
            S3 prefix for uploading TIFFs
        s3_bucket : str
            S3 bucket name
        s3_region : str
            AWS region
        cache_dir : str
            Local cache directory path

        Returns
        -------
        str
            S3 URI of uploaded file
        """
        from pathlib import Path

        import pooch

        from ocr.input_datasets.storage import S3Uploader

        # Download and extract using pooch
        cache_dir = get_temp_dir()
        cache_path = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir())
        cache_path.mkdir(parents=True, exist_ok=True)

        # Get URL and hash (must be defined for this dataset)
        url = var_info['url']
        if url is None:
            raise ValueError(f'URL not defined for variable {var_name}')
        hash_value = var_info.get('hash')  # Can be None

        extracted_files = pooch.retrieve(
            url=url,
            known_hash=hash_value,
            fname=f'{rds_id}-{var_name}-CONUS.zip',
            path=cache_path,
            processor=pooch.Unzip(),
            progressbar=True,
        )

        # Find the TIFF file
        if isinstance(extracted_files, str):
            extracted_files = [extracted_files]

        tiff_files = [f for f in extracted_files if f.endswith('.tif')]
        if not tiff_files:
            raise FileNotFoundError(f'No .tif files found for {var_name}')

        # Upload to S3
        tiff_path = Path(tiff_files[0])
        s3_key = f'{s3_tiff_prefix}/{var_name}_CONUS.tif'

        uploader = S3Uploader(s3_bucket, s3_region, dry_run=False)
        s3_uri = uploader.upload_file(tiff_path, s3_key)

        return s3_uri

    def download(self) -> None:
        """Download all variable TIFFs from USFS Box and upload to S3.

        Uses pooch for download + automatic unzipping, which provides:
        - Automatic caching (won't re-download if file exists)
        - Hash verification (when hashes are provided)
        - Built-in progress bars
        - Automatic ZIP extraction

        If use_coiled=True, runs downloads in parallel on Coiled workers.
        """
        console.log(f'Downloading {len(self.VARIABLES)} variables from USFS Box...')

        if self.use_coiled:
            console.log('Running downloads on Coiled cluster...')
            cluster = self.get_coiled_cluster()
            client = cluster.get_client()

            # Submit download tasks to Coiled workers
            futures = []
            for var_name, var_info in self.VARIABLES.items():
                future = client.submit(
                    self._download_and_upload_variable,
                    var_name=var_name,
                    var_info=var_info,
                    rds_id=self.rds_id,
                    s3_tiff_prefix=self.s3_tiff_prefix,
                    s3_bucket=self.config.s3_bucket,
                    s3_region=self.config.s3_region,
                )
                futures.append((var_name, future))

            # Wait for all downloads to complete
            for var_name, future in futures:
                try:
                    s3_uri = future.result()
                    console.log(f'✓ {var_name}: {s3_uri}')
                except Exception as e:
                    console.log(f'[bold red]✗ {var_name} failed: {e}[/bold red]')
                    raise

            console.log(f'Downloaded and uploaded all {len(self.VARIABLES)} variables on Coiled')
        else:
            # Local execution
            s3_uploader = S3Uploader(
                self.config.s3_bucket, self.config.s3_region, dry_run=self.dry_run
            )

            for var_name, var_info in self.VARIABLES.items():
                console.log(f'Processing variable: {var_name}')

                # Get URL and hash (must be defined for this dataset)
                url = var_info['url']
                if url is None:
                    raise ValueError(f'URL not defined for variable {var_name}')
                hash_value = var_info.get('hash')  # Can be None

                # Download and extract ZIP using pooch
                extracted_files = self.retrieve(
                    url=url,
                    known_hash=hash_value,
                    fname=f'{self.rds_id}-{var_name}-CONUS.zip',
                    processor=pooch.Unzip(),
                )

                # Find the TIFF file from extracted files
                if isinstance(extracted_files, str):
                    extracted_files = [extracted_files]

                tiff_files = [f for f in extracted_files if f.endswith('.tif')]
                if not tiff_files:
                    raise FileNotFoundError(
                        f'No .tif files found in extracted archive for {var_name}'
                    )

                # Upload TIFF to S3
                tiff_path = Path(tiff_files[0])
                s3_key = f'{self.s3_tiff_prefix}/{var_name}_CONUS.tif'
                s3_uploader.upload_file(tiff_path, s3_key)

            console.log(f'Downloaded and uploaded all {len(self.VARIABLES)} variables locally')

    def merge_tiffs_to_icechunk(self) -> None:
        """Merge all TIFF variables into a single Icechunk store (EPSG:5070)."""
        console.log('Merging TIFF variables into Icechunk store...')

        if self.use_coiled:
            client = self.get_coiled_cluster().get_client()
            console.log(f'Using Coiled cluster: {client}')

        # Build paths to S3 TIFFs
        fpath_dict = {
            var_name: f's3://{self.config.s3_bucket}/{self.s3_tiff_prefix}/{var_name}_CONUS.tif'
            for var_name in self.VARIABLES
        }

        # Load and merge all TIFFs
        console.log('Loading TIFFs from S3...')
        merge_ds = xr.merge(
            [
                xr.open_dataset(fpath, engine='rasterio')
                .squeeze()
                .drop_vars('band')
                .rename({'band_data': var_name})
                for var_name, fpath in fpath_dict.items()
            ],
            compat='override',
            join='override',
        )

        # Rechunk for optimal storage
        merge_ds = merge_ds.chunk(self.CHUNK_SIZES)
        merge_ds = dask.base.optimize(merge_ds)[0]

        console.log(f'Merged dataset shape: {dict(merge_ds.sizes)}')
        console.log(f'Variables: {list(merge_ds.data_vars)}')

        # Write to Icechunk
        writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=self.s3_icechunk_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )
        writer.write(merge_ds, 'Add all variables to store')

        console.log('Successfully merged TIFFs to Icechunk store')

    def reproject_to_4326(self) -> None:
        """Reproject the merged dataset from EPSG:5070 to EPSG:4326."""
        console.log('Reprojecting dataset from EPSG:5070 to EPSG:4326...')

        if self.use_coiled:
            client = self.get_coiled_cluster().get_client()
            console.log(f'Using Coiled cluster: {client}')

        # Load the EPSG:5070 dataset
        reader = IcechunkReader(
            bucket=self.config.s3_bucket,
            prefix=self.s3_icechunk_prefix,
            region=self.config.s3_region,
            debug=self.config.debug,
        )

        session = reader.repo.readonly_session('main')
        console.log('Loading dataset from Icechunk store...')
        rps_30 = xr.open_zarr(session.store).persist()

        # Downcast to float32 for efficiency
        for var in list(rps_30):
            rps_30[var] = rps_30[var].astype('float32')

        # Assign CRS and reproject
        console.log('Assigning EPSG:5070 CRS...')
        rps_30 = assign_crs(rps_30, crs='EPSG:5070')

        console.log('Reprojecting to EPSG:4326...')
        rps_30_4326 = xr_reproject(rps_30, how='EPSG:4326')

        # Sort and rechunk
        rps_30_4326 = rps_30_4326.sortby(['latitude', 'longitude']).chunk(
            self.REPROJECTED_CHUNK_SIZES
        )

        # Add metadata
        rps_30_4326.attrs = {
            'title': self.rds_id,
            'version': self.version,
            'data_source': self.source_url,
            'description': (
                f'{self.description}. '
                'This dataset was created by combining multiple source tif files '
                'and re-projecting from EPSG:5070 to EPSG:4326. '
                'It is stored in the Icechunk storage format.'
            ),
            'EPSG': '4326',
            'resolution': '30m',
        }

        console.log(f'Reprojected dataset shape: {dict(rps_30_4326.sizes)}')

        # Write to Icechunk
        writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=self.s3_icechunk_4326_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )
        writer.write(rps_30_4326, 'Reproject to EPSG:4326')

        console.log('Successfully reprojected to EPSG:4326')

    def process(self) -> None:
        """Run the complete processing pipeline: merge TIFFs and reproject."""
        self.merge_tiffs_to_icechunk()
        self.reproject_to_4326()

        # Clean up Coiled cluster if created
        if self._coiled_cluster is not None:
            console.log('Shutting down Coiled cluster...')
            self._coiled_cluster.shutdown()
