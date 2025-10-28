"""USFS Spatial datasets of probabilistic wildfire risk components for
the United States (270m) (3rd Edition) (Dillon et al. 2023, RDS-2016-0034-3).

This module handles the processing pipeline for the USFS Dillon et al., 2023 dataset:
1. Download ZIP archive
2. Extract and upload to S3
3.


Reference:
https://www.fs.usda.gov/rds/archive/products/RDS-2016-0034-3
"""

import tempfile
from typing import ClassVar

import coiled
import dask.base
import xarray as xr
from odc.geo.xr import assign_crs, xr_reproject

from ocr import catalog
from ocr.console import console
from ocr.input_datasets.base import BaseDatasetProcessor, InputDatasetConfig
from ocr.input_datasets.storage import IcechunkReader, IcechunkWriter, S3Uploader


class Dillon2023Processor(BaseDatasetProcessor):
    """Processor for USFS Dillon et al., 2023 wildfire risk dataset."""

    dataset_name: str = 'dillon-et-al-2023'
    dataset_type = 'tensor'
    description: str = (
        'USFS Spatial datasets of probabilistic wildfire risk components for the United States (270m) (3rd Edition) '
        '(Dillon et al. 2023, RDS-2016-0034-3)'
    )
    source_url: str = 'https://www.fs.usda.gov/rds/archive/catalog/RDS-2016-0034-3'
    version: str = '2023'
    rds_id: str = 'RDS-2016-0034-3'
    coiled_software: str | None = None

    COILED_WORKERS: ClassVar[int] = 2
    COILED_WORKER_VM: ClassVar[str] = 'm8g.16xlarge'
    COILED_SCHEDULER_VM: ClassVar[str] = 'm8g.2xlarge'

    CHUNK_SIZES: ClassVar[dict[str, int]] = {'y': 6000, 'x': 4500}
    REPROJECTED_CHUNK_SIZES: ClassVar[dict[str, int]] = {'latitude': 6000, 'longitude': 4500}
    ARCHIVE_URL: ClassVar[str] = (
        'https://www.fs.usda.gov/rds/archive/products/RDS-2016-0034-3/RDS-2016-0034-3.zip'
    )

    ARCHIVE_HASH: ClassVar[str] = 'b5313ad0ca2deac11b1284472731cc0aedea1bd3f0e92823e0d5356cb6dfff73'

    def __init__(
        self,
        config: InputDatasetConfig | None = None,
        *,
        dry_run: bool = False,
        use_coiled: bool = False,
    ):
        """Initialize the Dillon et al. 2023 processor."""
        super().__init__(dry_run=dry_run, config=config)
        self.use_coiled = use_coiled
        self._coiled_cluster = None

    def get_coiled_cluster(self):
        """Get or create a Coiled cluster for distributed processing."""
        if self._coiled_cluster is None:
            console.log('Creating Coiled cluster for distributed processing...')

            cluster_config = {
                'name': f'ocr-{self.dataset_name}',
                'region': self.config.s3_region,
                'n_workers': [self.COILED_WORKERS, self.COILED_WORKERS, self.COILED_WORKERS * 3],
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

    @property
    def s3_tiff_prefix(self) -> str:
        """S3 prefix for storing extracted TIFFs."""
        return f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/raw-input-tiffs'

    @property
    def s3_icechunk_prefix(self) -> str:
        """S3 prefix for storing Icechunk dataset."""
        return f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/processed-270m-5070.icechunk'

    @property
    def s3_icechunk_4326_prefix(self) -> str:
        """S3 prefix for storing reprojected Icechunk dataset (EPSG:4326)."""
        return (
            f'{self.config.base_prefix}/tensor/USFS/{self.dataset_name}/processed-30m-4326.icechunk'
        )

    @staticmethod
    def _download_and_upload_archive(
        archive_url: str,
        archive_hash: str | None,
        rds_id: str,
        s3_tiff_prefix: str,
        s3_bucket: str,
        s3_region: str,
    ) -> list[str]:
        """Download archive and upload all TIFFs to S3.

        This is a static method so it can be serialized and executed on Coiled workers.

        Returns
        -------
        list[str]
            List of S3 URIs
        """
        from pathlib import Path

        import pooch

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

        console.log(f'Downloaded and extracted archive: {extracted_files}')

        # mkdir -p RDS-2016-0034-3
        # unzip RDS-2016-0034-3.zip -d RDS-2016-0034-3/

        # Upload the tiffs to s3

        # aws s3 cp RDS-2016-0034-3/Data/I_FSim_CONUS_LF2020_270m/ s3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2016-0032-3/input_tif/ --recursive --exclude "*" --include "*.tif"

        # Find all TIFF files in the extracted files
        if isinstance(extracted_files, str):
            extracted_files = [extracted_files]
        console.log(f'Extracted files: {extracted_files}')

        tiff_files = [
            Path(f)
            for f in extracted_files
            if f.endswith('.tif') and 'Data/I_FSim_CONUS_LF2020_270m/' in f
        ]
        if not tiff_files:
            raise FileNotFoundError('No .tif files found in extracted archive')

        # Upload TIFFs to S3, preserving directory structure
        uploader = S3Uploader(s3_bucket, s3_region, dry_run=False)
        s3_uris = []

        for tiff_path in tiff_files:
            relative_path = tiff_path.relative_to(
                cache_path / f'{rds_id}.zip.unzip' / 'Data' / 'I_FSim_CONUS_LF2020_270m'
            )

            s3_key = f'{s3_tiff_prefix}/{relative_path.as_posix()}'
            console.log(f'Uploading {tiff_path} to S3 as {s3_key}')
            s3_uri = uploader.upload_file(tiff_path, s3_key)
            s3_uris.append(s3_uri)
        return s3_uris

    def download(self) -> None:
        """Download raw source data and upload to S3."""
        console.log(f'Downloading {self.rds_id} archive and uploading TIFFs to S3...')
        try:
            if self.use_coiled:
                cluster = self.get_coiled_cluster()
                client = cluster.get_client()
                with cluster:
                    future = client.submit(
                        self._download_and_upload_archive,
                        self.ARCHIVE_URL,
                        self.ARCHIVE_HASH,
                        self.rds_id,
                        self.s3_tiff_prefix,
                        self.config.s3_bucket,
                        self.config.s3_region,
                    )
                    s3_uris = future.result()
            else:
                s3_uris = self._download_and_upload_archive(
                    self.ARCHIVE_URL,
                    self.ARCHIVE_HASH,
                    self.rds_id,
                    self.s3_tiff_prefix,
                    self.config.s3_bucket,
                    self.config.s3_region,
                )
        except Exception as e:
            console.log(f'[bold red]✗ Error during download and upload: {e}[/bold red]')
            raise

        console.log(f'Uploaded {len(s3_uris)} TIFF files to S3 for dataset {self.dataset_name}')

    def merge_tiffs_to_icechunk(self) -> None:
        """Merge TIFF files from S3 into an Icechunk dataset and upload to S3."""
        console.log(f'Merging TIFF files into Icechunk dataset for {self.dataset_name}...')
        var_list = ['BP', 'FLP1', 'FLP2', 'FLP3', 'FLP4', 'FLP5', 'FLP6']
        fpath_dict = {
            var_name: f's3://{self.config.s3_bucket}/{self.s3_tiff_prefix}/CONUS_{var_name}.tif'
            for var_name in var_list
        }

        console.log(f'Opening datasets from S3: {fpath_dict}')
        # merge all the datasets and rename vars
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
        ).chunk(self.CHUNK_SIZES)

        merge_ds = dask.base.optimize(merge_ds)[0]
        console.log(f'Merged dataset: {merge_ds}')
        # Write to icechunk
        writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=self.s3_icechunk_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )
        writer.write(merge_ds, 'Add all raster data to store')
        console.log(
            f'Successfully wrote Icechunk dataset to s3://{self.config.s3_bucket}/{self.s3_icechunk_prefix}'
        )

    def reproject_to_4326(self) -> None:
        """Reproject the Icechunk dataset to EPSG:4326 and upload to S3."""
        console.log(f'Reprojecting Icechunk dataset to EPSG:4326 for {self.dataset_name}...')

        # Open the existing Icechunk dataset
        reader = IcechunkReader(
            bucket=self.config.s3_bucket,
            prefix=self.s3_icechunk_prefix,
            region=self.config.s3_region,
            debug=self.config.debug,
        )
        session = reader.repo.readonly_session('main')
        console.log('Loading dataset from Icechunk store...')

        ds = xr.open_dataset(session.store, engine='zarr', chunks={})
        for variable in ds.data_vars:
            ds[variable] = ds[variable].astype('float32')

        ds = assign_crs(ds, crs='EPSG:5070')
        console.log(f'Pre-reprojection dataset: {ds}')
        rps_30_4326 = catalog.get_dataset('scott-et-al-2024-30m-4326').to_xarray()
        rps_30_4326 = assign_crs(rps_30_4326, 'EPSG:4326')
        dset = (
            xr_reproject(ds, how=rps_30_4326.odc.geobox)
            .sortby(['latitude', 'longitude'])
            .chunk(self.REPROJECTED_CHUNK_SIZES)
        )
        # force the canonical coordinates from scott-et-al-2024 onto reprojected riley dataset
        dset = dset.assign_coords(longitude=rps_30_4326.longitude, latitude=rps_30_4326.latitude)
        dset = dask.base.optimize(dset)[0]
        console.log(f'Reprojected dataset: {dset}')

        dset.attrs = {
            'title': self.rds_id,
            'version': self.version,
            'data_source': self.source_url,
            'description': self.description,
            'EPSG': '4326',
            'resolution': '30m',
            'DOI': 'https://www.fs.usda.gov/rds/archive/catalog/RDS-2016-0034-3',
        }

        # Write reprojected dataset to new Icechunk
        writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=self.s3_icechunk_4326_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )
        writer.write(dset, 'Write reprojected Icechunk')
        console.log(
            f'Successfully wrote reprojected Icechunk dataset to s3://{self.config.s3_bucket}/{self.s3_icechunk_4326_prefix}'
        )

    def process(self) -> None:
        """Run the complete processing pipeline."""
        console.log(f'Processing dataset {self.dataset_name}...')
        if self.use_coiled:
            client = self.get_coiled_cluster().get_client()
            console.log(f'Using Coiled cluster: {client}')
        try:
            self.merge_tiffs_to_icechunk()
            self.reproject_to_4326()

        except Exception as e:
            console.log(f'[bold red]✗ Error during processing: {e}[/bold red]')
            raise

        # Clean up Coiled cluster if created
        if self._coiled_cluster is not None:
            console.log('Shutting down Coiled cluster...')
            self._coiled_cluster.shutdown()
