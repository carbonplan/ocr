from ocr.config import OCRConfig


def create_pyramid(config: OCRConfig):
    """
    Custom pyramid generation for the carbonplan/ocr project. For 30m CONUS raster data, standard methods of ndpyramid were breaking.
    This takes a slightly different approach:
    - Uses a single VM with the dask threaded scheduler to avoid huge distributed task graphs.
    - Creates an empty pyramid "template" / "skeleton" of the global extent, slippy-map tile shape web-mercator pyramid.
    - Trims the input data from a global extent to CONUS and then inserts it into the empty template/skeleton with xarray's region='auto'
    - Uses Xarray's coarsen to successively coarsen each higher level instead of regrinding each higher level.
    """

    import logging
    from dataclasses import dataclass, field
    from typing import Any

    import dask
    import dask.array
    import icechunk
    import morecantile
    import xarray as xr
    import zarr
    from dask.diagnostics import ProgressBar
    from numcodecs import Zlib
    from obstore.store import S3Store
    from odc.geo.geom import BoundingBox
    from odc.geo.xr import assign_crs, xr_reproject
    from zarr.storage import ObjectStore

    from ocr.console import console

    zarr.config.set({'async.concurrency': 128})
    dask.config.set(scheduler='threads', num_workers=64)

    logging.basicConfig(level=logging.INFO)
    # Silence tons of `INFO:botocore.credentials:Found credentials in environment variables.` info messages.
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)

    # -----------------------------------
    # ------------ utils ----------------
    # -----------------------------------

    @dataclass
    class PyramidConfig:
        level: int | None = None
        target_resolution: float | None = None
        pixels_per_tile: int = 512
        projection: str = 'EPSG:3857'
        projection_name: str = 'web-mercator'
        tms_id: str = 'WebMercatorQuad'
        resampling: str = 'average'
        compressor: Zlib = field(default_factory=lambda: Zlib(level=1))
        fill_value: float = 9.969209968386869e36
        regional_subset: BoundingBox | None = None
        dim: int = field(init=False)
        tms: morecantile.TileMatrixSet = field(init=False, repr=False)

        def __post_init__(self):
            self.tms = morecantile.tms.get(self.tms_id)

            if self.level is None and self.target_resolution is None:
                raise ValueError("Must specify either 'level' or 'target_resolution'")

            if self.level is not None and self.target_resolution is not None:
                raise ValueError("Specify only 'level' OR 'target_resolution', not both")

            if self.level is None:
                self.level = self.tms.zoom_for_res(self.target_resolution) - 1

            self.dim = 2**self.level * self.pixels_per_tile

        def get_affine(self):
            from affine import Affine

            return Affine.translation(-20037508.342789244, 20037508.342789248) * Affine.scale(
                (20037508.342789244 * 2) / self.dim, -(20037508.342789248 * 2) / self.dim
            )

        def get_geobox(self):
            from odc.geo.geobox import GeoBox

            return GeoBox((self.dim, self.dim), affine=self.get_affine(), crs=self.projection)

    def get_dataset_encoding(ds: xr.Dataset, config: PyramidConfig) -> dict[str, dict[str, Any]]:
        encoding = {}

        for var_name in ds.variables:
            var = ds[var_name]

            is_coordinate = var_name in ds.coords

            if is_coordinate:
                chunks = tuple(var.sizes[dim] for dim in var.dims)
            else:
                chunks = tuple(
                    config.pixels_per_tile if dim in ['x', 'y'] else var.sizes[dim]
                    for dim in var.dims
                )

            encoding[var_name] = {
                'compressor': config.compressor,
                'dtype': str(var.dtype),
                '_FillValue': config.fill_value,
                'chunks': chunks,
            }

        return encoding

    def get_multiscales_attrs(
        levels: list[int], config: PyramidConfig, is_datatree: bool = False
    ) -> dict[str, Any]:
        if is_datatree:
            datasets = [
                {
                    'path': str(level),
                    'level': level,
                    'crs': config.projection,
                    'pixels_per_tile': config.pixels_per_tile,
                }
                for level in levels
            ]
            attrs = {
                'multiscales': [{'datasets': datasets}],
                'title': 'multiscale data pyramid',
            }
        else:
            datasets = [{'path': '.', 'level': levels[0], 'crs': config.projection}]
            attrs = {'multiscales': [{'datasets': datasets}]}

        return attrs

    def create_skeleton_tree(
        datatree: xr.DataTree, config: PyramidConfig
    ) -> tuple[xr.DataTree, dict]:
        skeleton_tree = xr.DataTree()
        encodings = {}
        levels = list(range(config.level + 1))

        for group in levels:
            group_str = str(group)
            ds = datatree[group_str].ds

            group_ds = xr.Dataset(coords=ds.coords)

            for var in ds.data_vars:
                empty_array = dask.array.empty(ds[var].shape, dtype=ds[var].dtype, chunks=-1)
                group_ds[var] = xr.DataArray(empty_array, dims=ds[var].dims)

            group_ds.attrs = get_multiscales_attrs([group], config, is_datatree=False)
            skeleton_tree[group_str] = xr.DataTree(group_ds)

            encodings[f'/{group_str}'] = get_dataset_encoding(group_ds, config)

        skeleton_tree.attrs = get_multiscales_attrs(levels, config, is_datatree=True)

        return skeleton_tree, encodings

    def build_pyramid(ds: xr.Dataset, config: PyramidConfig) -> xr.DataTree:
        level_ds_dict = {str(config.level): ds}

        for lvl in range(config.level - 1, -1, -1):
            higher_ds = level_ds_dict[str(lvl + 1)]
            lower_ds = higher_ds.coarsen(x=2, y=2).mean()
            level_ds_dict[str(lvl)] = lower_ds

        datatree = xr.DataTree.from_dict(level_ds_dict)

        return datatree

    def write_pyramid_groups(
        datatree: xr.DataTree,
        config: PyramidConfig,
        output_bucket: str,
        output_prefix: str,
        region: str = 'us-west-2',
    ):
        for group in range(config.level, -1, -1):
            if config.debug:
                console.log(f'Writing pyramid group/level: {group}')
            group_str = str(group)

            batch_store = S3Store(
                output_bucket,
                prefix=f'{output_prefix}/{group_str}',
                region=region,
            )
            batch_zstore = ObjectStore(batch_store)

            subset_ds = datatree[group_str].ds

            if config.regional_subset:
                subset_ds = subset_ds.sel(
                    x=slice(config.regional_subset.left, config.regional_subset.right),
                    y=slice(config.regional_subset.top, config.regional_subset.bottom),
                )

            # encoding = get_dataset_encoding(subset_ds, config)
            subset_ds.attrs = get_multiscales_attrs([group], config, is_datatree=False)

            with ProgressBar():
                subset_ds.chunk({'x': config.pixels_per_tile, 'y': config.pixels_per_tile}).to_zarr(
                    batch_zstore,
                    consolidated=False,
                    align_chunks=True,
                    region='auto',
                    mode='r+',
                )

    # =========================================
    # ===========   CONFIG    =================
    # =========================================

    # BBOX subset for CONUS extent
    # TODO: For QA runs, our bbox will be way smaller.
    # A PERFORMANT way to get the bounds of the valid data for the template would be great.

    # REGIONAL_SUBSET = BoundingBox(
    #     left=-13927438.0498,
    #     bottom=2824695.1999,
    #     right=-7430902.1418,
    #     top=6280174.6103,
    #     crs='EPSG:3857',
    # )

    # !!!!!!!!!!!!!!!
    # TEMP SUBSET OF CENTRAL/WESTERN WA FOR TESTING OF REGION Y2_X5
    REGIONAL_SUBSET = BoundingBox(
        left=-13539749.4423,
        bottom=5824773.8598,
        right=-13028538.5972,
        top=6272621.1426,
        crs='EPSG:3857',
    )
    # !!!!!!!!!!!!!!!

    TARGET_RESOLUTION = 30
    PIXELS_PER_TILE = 512
    AWS_REGION = 'us-west-2'
    INPUT_RASTER_BUCKET = config.icechunk.storage_root
    INPUT_RASTER_PREFIX = config.icechunk.prefix

    PYRAMID_BUCKET = config.pyramid.storage_root
    PYRAMID_PREFIX = config.pyramid.prefix
    # VAR_LIST = ['wind_risk_2011', 'wind_risk_2047']

    config = PyramidConfig(
        target_resolution=TARGET_RESOLUTION,
        regional_subset=REGIONAL_SUBSET,
        pixels_per_tile=PIXELS_PER_TILE,
    )

    # =========================================
    # ===========   PROCESS    =================
    # =========================================

    storage = icechunk.s3_storage(
        bucket=INPUT_RASTER_BUCKET, prefix=INPUT_RASTER_PREFIX, from_env=True
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session('main')
    ds = xr.open_zarr(session.store, consolidated=False)  # [VAR_LIST]
    ds = assign_crs(ds, 'EPSG:4326')

    dst_geobox = config.get_geobox()
    odc_ds = xr_reproject(ds, dst_geobox, resampling=config.resampling)
    regridded_ds = assign_crs(odc_ds, config.projection)

    highest_level_ds = regridded_ds.drop_vars('spatial_ref')

    datatree = build_pyramid(highest_level_ds, config)
    skeleton_tree, skeleton_encodings = create_skeleton_tree(datatree, config)

    template_store = S3Store(
        PYRAMID_BUCKET,
        prefix=PYRAMID_PREFIX,
        region=AWS_REGION,
    )
    zstore_template = ObjectStore(template_store)

    skeleton_tree.to_zarr(
        zstore_template,
        encoding=skeleton_encodings,
        consolidated=False,
        compute=False,
        zarr_format=2,
        mode='w',
    )

    write_pyramid_groups(
        datatree=datatree, config=config, output_bucket=PYRAMID_BUCKET, output_prefix=PYRAMID_PREFIX
    )

    zarr.consolidate_metadata(zstore_template)
