"""CONUS404 hourly meteorological dataset processors.

This module handles processing pipelines for CONUS404 data:
1. Conus404SubsetProcessor: Download and rechunk CONUS404 variables from OSN
2. Conus404FFWIProcessor: Compute Fosberg Fire Weather Index with quantiles and reprojection

Reference:
https://www.usgs.gov/publications/conus404-a-high-resolution-reanalysis-dataset-north-america
OSN Storage: s3://hytest/conus404/conus404_hourly.zarr
"""

import time
from typing import ClassVar, Literal

import coiled
import dask.base
import xarray as xr
from odc.geo import CRS
from odc.geo.xr import assign_crs, xr_reproject

from ocr import catalog
from ocr.console import console
from ocr.conus404 import (
    compute_relative_humidity,
    compute_wind_speed_and_direction,
    load_conus404,
    rotate_winds_to_earth,
)
from ocr.input_datasets.base import BaseDatasetProcessor, InputDatasetConfig
from ocr.input_datasets.storage import IcechunkReader, IcechunkWriter
from ocr.risks.fire import fosberg_fire_weather_index


class Conus404SubsetProcessor(BaseDatasetProcessor):
    """Processor for CONUS404 variable subset from Open Storage Network (OSN).

    Downloads specific meteorological variables from CONUS404 hosted on OSN
    and rechunks them from temporal+spatial chunks to spatial-only chunks
    for efficient spatial queries.
    """

    # Dataset metadata
    dataset_name: str = 'conus404-subset'
    dataset_type = 'tensor'
    description: str = (
        'CONUS-404 hourly meteorological variables rechunked from Open Storage Network'
    )
    source_url: str = 's3://hytest/conus404/conus404_hourly.zarr'
    version: str = 'hourly'

    # Available variables for download
    VARIABLES: ClassVar[list[str]] = ['Q2', 'TD2', 'PSFC', 'T2', 'V10', 'U10']

    # Processing configuration
    COILED_WORKERS: ClassVar[int] = 15
    COILED_WORKER_VM: ClassVar[str] = 'r6a.8xlarge'
    COILED_SCHEDULER_VM: ClassVar[str] = 'c6a.large'
    SPATIAL_TILE_SIZE: ClassVar[int] = 10

    # OSN endpoint configuration
    OSN_ENDPOINT: ClassVar[str] = 'https://usgs.osn.mghpcc.org/'

    def __init__(
        self,
        config: InputDatasetConfig | None = None,
        *,
        dry_run: bool = False,
        use_coiled: bool = True,
        coiled_software: str | None = None,
        variable: str = 'Q2',
        spatial_tile_size: int | None = None,
    ):
        """Initialize the CONUS404 subset processor.

        Parameters
        ----------
        config : InputDatasetConfig, optional
            Configuration object. Creates default if None.
        dry_run : bool, default False
            If True, skip actual S3 writes and only log operations.
        use_coiled : bool, default True
            If True, use Coiled for distributed processing.
        coiled_software : str, optional
            Coiled software environment ID.
        variable : str, default 'Q2'
            Variable to process. Must be one of VARIABLES.
        spatial_tile_size : int, optional
            Size of spatial tiles for chunking. Uses class default if None.
        """
        super().__init__(config, dry_run=dry_run)
        self.use_coiled = use_coiled
        self.coiled_software = coiled_software
        self._coiled_cluster = None

        # Validate variable
        if variable not in self.VARIABLES:
            raise ValueError(f"Invalid variable '{variable}'. Must be one of {self.VARIABLES}")
        self.variable = variable
        self.spatial_tile_size = spatial_tile_size or self.SPATIAL_TILE_SIZE

    @property
    def s3_icechunk_prefix(self) -> str:
        """S3 prefix for variable-specific Icechunk store."""
        return f'input/conus404-hourly-icechunk/{self.variable}'

    def get_coiled_cluster(self):
        """Get or create a Coiled cluster for distributed processing."""
        if self._coiled_cluster is None:
            console.log('Creating Coiled cluster for CONUS404 processing...')
            self._coiled_cluster = coiled.Cluster(
                name=f'ocr-{self.dataset_name}-{self.variable}',
                region=self.config.s3_region,
                n_workers=[self.COILED_WORKERS, self.COILED_WORKERS, self.COILED_WORKERS],
                tags={'Project': 'OCR'},
                worker_vm_types=[self.COILED_WORKER_VM],
                scheduler_vm_types=[self.COILED_SCHEDULER_VM],
                software=self.coiled_software,
                spot_policy='spot_with_fallback',
            )
            console.log(f'Coiled cluster created: {self._coiled_cluster.dashboard_link or "N/A"}')
        return self._coiled_cluster

    @staticmethod
    def _load_and_rechunk(
        variable: str,
        spatial_tile_size: int,
        osn_endpoint: str,
    ) -> xr.Dataset:
        """Load and rechunk a CONUS404 variable from OSN.

        This static method can be serialized and sent to Coiled workers.

        Parameters
        ----------
        variable : str
            Variable name to load.
        spatial_tile_size : int
            Size of spatial tiles for chunking.
        osn_endpoint : str
            OSN endpoint URL.

        Returns
        -------
        xr.Dataset
            Rechunked dataset with time: -1, x: spatial_tile_size, y: spatial_tile_size.
        """
        import xarray as xr
        from distributed import wait

        ds = xr.open_zarr(
            's3://hytest/conus404/conus404_hourly.zarr',
            storage_options={
                'anon': True,
                'client_kwargs': {'endpoint_url': osn_endpoint},
            },
        )[[variable]]

        # Rechunk from temporal+spatial to spatial-only chunks
        ds = ds.chunk({'time': -1, 'x': spatial_tile_size, 'y': spatial_tile_size}).persist()
        wait(ds)

        # Clean up encoding to avoid conflicts
        for var in ds.variables:
            ds[var].encoding.pop('chunks', None)
            ds[var].encoding.pop('preferred_chunks', None)
            ds[var].encoding.pop('compressors', None)

        return ds

    def download(self) -> None:
        """Download step (no-op for CONUS404).

        Data is accessed directly from OSN during the process step.
        """
        console.log(
            f'Skipping download - {self.variable} will be read directly from OSN during processing'
        )

    def process(self) -> None:
        """Process CONUS404 variable: load from OSN, rechunk, and write to Icechunk."""
        start_time = time.time()
        console.log(f'Processing CONUS404 variable: {self.variable}')
        console.log(f'Spatial tile size: {self.spatial_tile_size}')

        if self.dry_run:
            console.log('[DRY RUN] Would process the following:')
            console.log(f'  - Variable: {self.variable}')
            console.log(f'  - Source: {self.source_url}')
            console.log(f'  - OSN endpoint: {self.OSN_ENDPOINT}')
            console.log(f'  - Spatial tile size: {self.spatial_tile_size}')
            console.log(f'  - Output: s3://{self.config.s3_bucket}/{self.s3_icechunk_prefix}')
            console.log('[DRY RUN] Skipping actual data loading and processing')
            return

        if self.use_coiled:
            cluster = self.get_coiled_cluster()
            client = cluster.get_client()
            console.log(f'Using Coiled cluster: {client}')
            console.log('Waiting 30 seconds for cluster to stabilize...')
            time.sleep(30)

        # Load and rechunk dataset
        ds = self._load_and_rechunk(
            self.variable,
            self.spatial_tile_size,
            self.OSN_ENDPOINT,
        )

        console.log(f'Loaded and rechunked dataset: {ds}')

        # Write to Icechunk
        writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=self.s3_icechunk_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )

        snapshot_id = writer.write(
            ds,
            commit_message=f'Add {self.variable} from CONUS404 OSN with spatial tile size {self.spatial_tile_size}',
        )

        if not self.dry_run:
            console.log(f'Written to Icechunk snapshot: {snapshot_id}')

        # Cleanup
        if self._coiled_cluster is not None:
            console.log('Shutting down Coiled cluster...')
            self._coiled_cluster.shutdown()

        elapsed = time.time() - start_time
        console.log(f'Processing completed in {elapsed / 60:.2f} minutes')


class Conus404FFWIProcessor(BaseDatasetProcessor):
    """Processor for Fosberg Fire Weather Index (FFWI) derived from CONUS404.

    Computes FFWI from CONUS404 meteorological variables with three processing steps:
    1. compute: Calculate base FFWI and winds from relative humidity, temperature, wind speed
    2. postprocess: Compute quantiles (e.g., 99th percentile) over time dimension
    3. reproject: Reproject wind direction distribution to EPSG:4326 geobox
    """

    # Dataset metadata
    dataset_name: str = 'conus404-ffwi'
    dataset_type = 'tensor'
    description: str = (
        'Fosberg Fire Weather Index computed from CONUS404 hourly meteorological data'
    )
    source_url: str = 's3://hytest/conus404/conus404_hourly.zarr'
    version: str = 'v1'

    # Processing configuration
    COILED_WORKERS_COMPUTE: ClassVar[int] = 20
    COILED_WORKERS_POSTPROCESS: ClassVar[int] = 20
    COILED_WORKERS_REPROJECT: ClassVar[int] = 20
    COILED_WORKER_VM: ClassVar[str] = 'm8g.4xlarge'
    COILED_SCHEDULER_VM: ClassVar[str] = 'm8g.2xlarge'

    # Chunk sizes for different stages
    CHUNK_SIZES_COMPUTE: ClassVar[dict[str, int]] = {'x': 10, 'y': 10}
    CHUNK_SIZES_QUANTILE: ClassVar[dict[str, int]] = {'x': -1, 'y': -1}
    CHUNK_SIZES_REPROJECT: ClassVar[dict[str, int]] = {'latitude': 6000, 'longitude': 4500}

    # Output base path
    OUTPUT_BASE_PREFIX: ClassVar[str] = 'input/fire-risk/tensor/conus404'

    def __init__(
        self,
        config: InputDatasetConfig | None = None,
        *,
        dry_run: bool = False,
        use_coiled: bool = True,
        coiled_software: str | None = None,
        processing_step: Literal['compute', 'postprocess', 'reproject', 'all'] = 'all',
        quantiles: list[float] | None = None,
    ):
        """Initialize the CONUS404 FFWI processor.

        Parameters
        ----------
        config : InputDatasetConfig, optional
            Configuration object. Creates default if None.
        dry_run : bool, default False
            If True, skip actual S3 writes and only log operations.
        use_coiled : bool, default True
            If True, use Coiled for distributed processing.
        coiled_software : str, optional
            Coiled software environment ID.
        processing_step : {'compute', 'postprocess', 'reproject', 'all'}, default 'all'
            Which processing step(s) to execute.
        quantiles : list[float], optional
            Quantiles to compute in postprocess step. Default is [0.99].
        """
        super().__init__(config, dry_run=dry_run)
        self.use_coiled = use_coiled
        self.coiled_software = coiled_software
        self.processing_step = processing_step
        self.quantiles = quantiles or [0.99]
        self._coiled_cluster = None

    @property
    def s3_ffwi_prefix(self) -> str:
        """S3 prefix for base FFWI Icechunk store."""
        return f'{self.OUTPUT_BASE_PREFIX}/fosberg-fire-weather-index.icechunk'

    @property
    def s3_winds_prefix(self) -> str:
        """S3 prefix for winds Icechunk store."""
        return f'{self.OUTPUT_BASE_PREFIX}/winds.icechunk'

    def s3_quantile_prefix(self, quantile: float) -> str:
        """S3 prefix for quantile-specific Icechunk store."""
        return (
            f'{self.OUTPUT_BASE_PREFIX}/fosberg-fire-weather-index-p{int(quantile * 100)}.icechunk'
        )

    @property
    def s3_distribution_prefix(self) -> str:
        """S3 prefix for wind direction distribution Icechunk store."""
        return f'{self.OUTPUT_BASE_PREFIX}/fosberg-fire-weather-index-p99-wind-direction-distribution.icechunk'

    @property
    def s3_distribution_reprojected_prefix(self) -> str:
        """S3 prefix for reprojected wind direction distribution."""
        return f'{self.OUTPUT_BASE_PREFIX}/fosberg-fire-weather-index-p99-wind-direction-distribution-30m-4326.icechunk'

    def get_coiled_cluster(self, step: str = 'compute'):
        """Get or create a Coiled cluster for distributed processing.

        Parameters
        ----------
        step : str, default 'compute'
            Processing step name, used to determine worker count.

        Returns
        -------
        coiled.Cluster
            Coiled cluster instance.
        """
        if self._coiled_cluster is None:
            # Determine worker count based on step
            worker_counts = {
                'compute': self.COILED_WORKERS_COMPUTE,
                'postprocess': self.COILED_WORKERS_POSTPROCESS,
                'reproject': self.COILED_WORKERS_REPROJECT,
            }
            n_workers = worker_counts.get(step, self.COILED_WORKERS_COMPUTE)

            console.log(f'Creating Coiled cluster for FFWI {step} step...')
            self._coiled_cluster = coiled.Cluster(
                name=f'ocr-{self.dataset_name}-{step}',
                region=self.config.s3_region,
                n_workers=[n_workers * 2, n_workers * 2, n_workers * 4],
                tags={'Project': 'OCR'},
                worker_vm_types=[self.COILED_WORKER_VM],
                scheduler_vm_types=[self.COILED_SCHEDULER_VM],
                software=self.coiled_software,
                idle_timeout='10 minutes',
                spot_policy='spot_with_fallback',
            )
            console.log(f'Coiled cluster created: {self._coiled_cluster.dashboard_link or "N/A"}')
        return self._coiled_cluster

    def _validate_prerequisite(self, prefix: str, step_name: str) -> None:
        """Validate that prerequisite data exists before processing.

        Parameters
        ----------
        prefix : str
            S3 prefix to check for existence.
        step_name : str
            Name of the step requiring this prerequisite.

        Raises
        ------
        FileNotFoundError
            If prerequisite data does not exist.
        """
        if self.dry_run:
            console.log(f'[dry-run] Skipping prerequisite validation for {step_name}')
            return

        try:
            reader = IcechunkReader(
                bucket=self.config.s3_bucket,
                prefix=prefix,
                region=self.config.s3_region,
            )
            # Try to access the repository
            _ = reader.repo.readonly_session('main')
            console.log(f'✓ Prerequisite data found for {step_name}: {prefix}')
        except Exception as e:
            raise FileNotFoundError(
                f'Prerequisite data not found for {step_name}. '
                f'Expected Icechunk repository at s3://{self.config.s3_bucket}/{prefix}. '
                f'Run with --ffwi-processing-step=compute first. Error: {e}'
            )

    def _reproject(
        self,
        src_dataset: xr.Dataset,
        src_crs_wkt: str,
    ) -> xr.Dataset:
        """Reproject dataset to the geobox of the target dataset.

        Parameters
        ----------
        src_dataset : xr.Dataset
            Source dataset to reproject.
        src_crs_wkt : str
            WKT string of source CRS.

        Returns
        -------
        xr.Dataset
            Reprojected dataset.
        """
        target_dataset_name = 'scott-et-al-2024-30m-4326'
        tgt = catalog.get_dataset(target_dataset_name).to_xarray().astype('float32')
        tgt = assign_crs(tgt, crs='EPSG:4326')
        geobox = tgt.odc.geobox

        src_crs = CRS(src_crs_wkt)
        src_dataset = assign_crs(src_dataset, crs=src_crs)
        result = xr_reproject(
            src_dataset,
            geobox,
            resampling='nearest',
        ).astype('float32')

        # Fix coordinate floating point noise (see https://github.com/carbonplan/ocr/issues/247)
        result = result.assign_coords(latitude=tgt.latitude, longitude=tgt.longitude)

        # Sort coordinates and chunk
        result = result.sortby(['latitude', 'longitude']).chunk(self.CHUNK_SIZES_REPROJECT)
        result.attrs.update({'reprojected_to': target_dataset_name})
        return result

    def _compute_base_ffwi(self) -> None:
        """Compute base FFWI and winds from CONUS404 data."""
        console.log('Step 1: Computing base FFWI and winds...')
        start_time = time.time()

        if self.dry_run:
            console.log('[DRY RUN] Would compute base FFWI with the following:')
            console.log(f'  - Source: {self.source_url}')
            console.log(f'  - FFWI output: s3://{self.config.s3_bucket}/{self.s3_ffwi_prefix}')
            console.log(f'  - Winds output: s3://{self.config.s3_bucket}/{self.s3_winds_prefix}')
            console.log(f'  - Chunk sizes: {self.CHUNK_SIZES_COMPUTE}')
            console.log('[DRY RUN] Skipping actual FFWI computation')
            return

        if self.use_coiled:
            cluster = self.get_coiled_cluster('compute')
            client = cluster.get_client()
            console.log(f'Using Coiled cluster: {client}')
            console.log('Waiting 60 seconds for cluster to stabilize...')
            time.sleep(60)

        # Load CONUS404 data
        ds = load_conus404(add_spatial_constants=True)
        console.log(f'Loaded CONUS404 data: {ds}')

        # Compute relative humidity
        console.log('Computing relative humidity...')
        hurs = compute_relative_humidity(ds)

        # Rotate winds to earth coordinates
        console.log('Rotating winds to earth coordinates...')
        earth_u, earth_v = rotate_winds_to_earth(ds)
        wind_ds = compute_wind_speed_and_direction(earth_u, earth_v)

        # Compute FFWI
        console.log('Computing Fosberg Fire Weather Index...')
        ffwi = fosberg_fire_weather_index(
            hurs=hurs, T2=ds['T2'], sfcWind=wind_ds['sfcWind']
        ).to_dataset()
        ffwi = ffwi.chunk(self.CHUNK_SIZES_COMPUTE).persist()

        console.log(f'Computed FFWI: {ffwi}')

        # Write FFWI
        writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=self.s3_ffwi_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )
        snapshot_id = writer.write(
            ffwi,
            commit_message='Add Fosberg Fire Weather Index data.',
        )
        if not self.dry_run:
            console.log(f'FFWI written to snapshot: {snapshot_id}')

        # Write winds
        wind_ds = wind_ds.chunk(self.CHUNK_SIZES_COMPUTE)
        wind_writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=self.s3_winds_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )
        wind_snapshot_id = wind_writer.write(
            wind_ds,
            commit_message='Add surface wind data used in Fosberg Fire Weather Index computation.',
        )
        if not self.dry_run:
            console.log(f'Winds written to snapshot: {wind_snapshot_id}')

        elapsed = time.time() - start_time
        console.log(f'Base FFWI computation completed in {elapsed / 60:.2f} minutes')

    def _compute_quantiles(self) -> None:
        """Compute quantiles from base FFWI data."""
        console.log('Step 2: Computing quantiles from base FFWI...')
        start_time = time.time()

        # Validate prerequisite
        self._validate_prerequisite(self.s3_ffwi_prefix, 'postprocess')

        if self.use_coiled:
            cluster = self.get_coiled_cluster('postprocess')
            client = cluster.get_client()
            console.log(f'Using Coiled cluster: {client}')
            console.log('Waiting 60 seconds for cluster to stabilize...')
            time.sleep(60)

        # Load base FFWI
        console.log(f'Loading base FFWI from {self.s3_ffwi_prefix}...')
        reader = IcechunkReader(
            bucket=self.config.s3_bucket,
            prefix=self.s3_ffwi_prefix,
            region=self.config.s3_region,
        )
        session = reader.repo.readonly_session('main')
        ffwi = xr.open_zarr(session.store)
        console.log(f'Loaded FFWI: {ffwi}')

        # Compute and save each quantile
        for quantile in self.quantiles:
            console.log(f'Computing {quantile} quantile...')
            ffwi_quantile = ffwi.quantile(quantile, dim='time').chunk(self.CHUNK_SIZES_QUANTILE)
            ffwi_quantile = dask.base.optimize(ffwi_quantile)[0]

            ffwi_quantile.FFWI.attrs['description'] = (
                f'Fosberg Fire Weather Index {quantile} quantile over time dimension.'
            )
            ffwi_quantile.FFWI.attrs['computed_for_quantile'] = quantile

            # Write quantile
            q_writer = IcechunkWriter(
                bucket=self.config.s3_bucket,
                prefix=self.s3_quantile_prefix(quantile),
                region=self.config.s3_region,
                dry_run=self.dry_run,
                debug=self.config.debug,
            )
            q_snapshot_id = q_writer.write(
                ffwi_quantile,
                commit_message=f'Add Fosberg Fire Weather Index {quantile} quantile data.',
            )
            if not self.dry_run:
                console.log(f'Quantile {quantile} written to snapshot: {q_snapshot_id}')

        # Compute wind direction distribution for the highest quantile (typically 99th percentile)
        if self.quantiles:
            highest_quantile = max(self.quantiles)
            console.log(
                f'Computing wind direction distribution for {highest_quantile} quantile FFWI..'
            )

            # Load winds dataset
            console.log(f'Loading winds from {self.s3_winds_prefix}...')
            winds_reader = IcechunkReader(
                bucket=self.config.s3_bucket,
                prefix=self.s3_winds_prefix,
                region=self.config.s3_region,
            )
            winds_session = winds_reader.repo.readonly_session('main')
            winds = xr.open_zarr(winds_session.store)

            # Create fire weather mask: FFWI >= quantile threshold
            ffwi_threshold = ffwi.FFWI.quantile(highest_quantile, dim='time')
            fire_weather_mask = ffwi.FFWI >= ffwi_threshold

            # Compute wind direction distribution using fire.py function
            from ocr.risks.fire import compute_wind_direction_distribution

            wind_direction_dist = compute_wind_direction_distribution(
                direction=winds.sfcWindfromdir,
                fire_weather_mask=fire_weather_mask,
            )

            # Write wind direction distribution
            dist_writer = IcechunkWriter(
                bucket=self.config.s3_bucket,
                prefix=self.s3_distribution_prefix,
                region=self.config.s3_region,
                dry_run=self.dry_run,
                debug=self.config.debug,
            )
            dist_snapshot_id = dist_writer.write(
                wind_direction_dist,
                commit_message=f'Add wind direction distribution for FFWI >= {highest_quantile} quantile.',
            )
            if not self.dry_run:
                console.log(f'Wind direction distribution written to snapshot: {dist_snapshot_id}')

        elapsed = time.time() - start_time
        console.log(f'Quantile computation completed in {elapsed / 60:.2f} minutes')

    def _reproject_distribution(self) -> None:
        """Reproject wind direction distribution to EPSG:4326."""
        console.log('Step 3: Reprojecting wind direction distribution...')
        start_time = time.time()

        # Validate prerequisite (check if distribution exists)
        # Note: In the original script, this is computed in postprocess step
        # For now, we'll just validate that it exists
        self._validate_prerequisite(
            self.s3_distribution_prefix, 'reproject (wind direction distribution)'
        )

        if self.use_coiled:
            cluster = self.get_coiled_cluster('reproject')
            client = cluster.get_client()
            console.log(f'Using Coiled cluster: {client}')
            console.log('Waiting 60 seconds for cluster to stabilize...')
            time.sleep(60)

        # Load wind direction distribution
        console.log(f'Loading wind direction distribution from {self.s3_distribution_prefix}...')
        reader = IcechunkReader(
            bucket=self.config.s3_bucket,
            prefix=self.s3_distribution_prefix,
            region=self.config.s3_region,
        )
        session = reader.repo.readonly_session('main')
        wind_dir_dist = xr.open_zarr(session.store)
        console.log(f'Loaded wind direction distribution: {wind_dir_dist}')

        # Get CONUS404 CRS
        conus404 = load_conus404(add_spatial_constants=True)
        src_crs_wkt = conus404['crs'].attrs['crs_wkt']
        console.log(f'CONUS404 CRS WKT: {src_crs_wkt[:100]}...')

        # Reproject
        console.log('Reprojecting to EPSG:4326...')
        reprojected = self._reproject(wind_dir_dist, src_crs_wkt)
        console.log(f'Reprojected distribution: {reprojected}')

        # Write reprojected data
        reproject_writer = IcechunkWriter(
            bucket=self.config.s3_bucket,
            prefix=self.s3_distribution_reprojected_prefix,
            region=self.config.s3_region,
            dry_run=self.dry_run,
            debug=self.config.debug,
        )
        snapshot_id = reproject_writer.write(
            reprojected,
            commit_message='Reprojected wind direction distribution to 30m EPSG:4326 geobox of Scott et al. 2024 dataset.',
        )
        if not self.dry_run:
            console.log(f'Reprojected data written to snapshot: {snapshot_id}')

        elapsed = time.time() - start_time
        console.log(f'Reprojection completed in {elapsed / 60:.2f} minutes')

    def download(self) -> None:
        """Download step (no-op for FFWI).

        Data is computed from CONUS404, not downloaded from external source.
        """
        console.log('Skipping download - FFWI is computed from existing CONUS404 data')

    def process(self) -> None:
        """Process FFWI: compute, postprocess quantiles, and/or reproject based on step parameter."""
        console.log(f'Processing FFWI with step(s): {self.processing_step}')

        steps_to_run = []
        if self.processing_step == 'all':
            steps_to_run = ['compute', 'postprocess', 'reproject']
        else:
            steps_to_run = [self.processing_step]

        for step in steps_to_run:
            if step == 'compute':
                self._compute_base_ffwi()
            elif step == 'postprocess':
                self._compute_quantiles()
            elif step == 'reproject':
                self._reproject_distribution()

        # Cleanup
        if self._coiled_cluster is not None:
            console.log('Shutting down Coiled cluster...')
            self._coiled_cluster.shutdown()

        console.log('FFWI processing complete')
