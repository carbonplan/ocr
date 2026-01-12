"""
California Fire Hazard Severity Zones (FHSZ) dataset processor.
"""

import coiled
import dask.base
import xarray as xr
from odc.geo.xr import assign_crs, xr_reproject

from ocr.console import console
from ocr.input_datasets.base import BaseDatasetProcessor, InputDatasetConfig
from ocr.input_datasets.storage import IcechunkReader, IcechunkWriter


class CalfireFHSZProcessor(BaseDatasetProcessor):
    dataset_name = 'calfire-fhsz'
    dataset_type = 'tensor'
    description = 'California Fire Hazard Severity Zones (FHSZ) dataset from CAL FIRE'

    COILED_SOFTWARE = None
    COILED_WORKERS = 2
    COILED_WORKER_VM = 'm8g.2xlarge'
    COILED_SCHEDULER_VM = 'm8g.2xlarge'

    CHUNK_SIZES = {'y': 6000, 'x': 4500}
    REPROJECTED_CHUNK_SIZES = {'latitude': 6000, 'longitude': 4500}
    TIFF_URL = 's3://carbonplan-ocr/input/fire-risk/tensor/calfire-fhsz/calfire-risk-raster.tif'

    def __init__(
        self,
        config: InputDatasetConfig | None = None,
        *,
        dry_run: bool = False,
        use_coiled: bool = False,
    ):
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
                'software': self.COILED_SOFTWARE,
                'wait_for_workers': 1,
                'idle_timeout': '5 minutes',
                'worker_disk_size': 150,
            }

            self._coiled_cluster = coiled.Cluster(**cluster_config)

            console.log(f'Coiled cluster created: {self._coiled_cluster.dashboard_link or "N/A"}')
        return self._coiled_cluster

    def download(self) -> None:
        """Download is not needed - data is downloaded directly during processing."""
        console.log('Skipping download - data is downloaded during processing')

    @property
    def s3_icechunk_prefix(self) -> str:
        return f'{self.config.base_prefix}/tensor/{self.dataset_name}/calfire-risk-raster-3310.icechunk'

    @property
    def s3_icechunk_4326_prefix(self) -> str:
        return f'{self.config.base_prefix}/tensor/{self.dataset_name}/calfire-risk-raster-4326.icechunk'

    def tiff_to_icechunk(self) -> None:
        """Convert the TIFF to Icechunk format, optionally using Coiled."""
        console.log(f'Starting TIFF: {self.TIFF_URL} to Icechunk conversion...')
        ds = xr.open_dataset(self.TIFF_URL, engine='rasterio', chunks={})
        ds = ds.odc.assign_crs('epsg:3310').chunk({'y': 6000, 'x': 4500})

        writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=self.s3_icechunk_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )
        writer.write(ds, 'Adding FHSZ data')
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

        ds = assign_crs(ds, crs='EPSG:3310')
        console.log(f'Pre-reprojection dataset: {ds}')

        dset = (
            xr_reproject(ds, how='EPSG: 4326')
            .sortby(['latitude', 'longitude'])
            .chunk(self.REPROJECTED_CHUNK_SIZES)
        )

        dset = dask.base.optimize(dset)[0]
        console.log(f'Reprojected dataset: {dset}')

        dset.attrs = {
            'title': 'California Fire Hazard Severity Zones (FHSZ)',
            'version': self.version,
            'data_source': self.source_url,
            'description': self.description,
            'EPSG': '4326',
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
            # self.tiff_to_icechunk()
            self.reproject_to_4326()

        except Exception as e:
            console.log(f'[bold red]✗ Error during processing: {e}[/bold red]')
            raise

        # Clean up Coiled cluster if created
        if self._coiled_cluster is not None:
            console.log('Shutting down Coiled cluster...')
            self._coiled_cluster.shutdown()
