from ocr.config import OCRConfig


def create_pyramid(config: OCRConfig):
    import logging

    import dask
    import dask.array
    import icechunk
    import morecantile
    import xarray as xr
    import zarr
    from obstore.store import from_url
    from topozarr.coarsen import create_pyramid
    from zarr.storage import ObjectStore

    zarr.config.set({'async.concurrency': 128})
    dask.config.set(scheduler='threads', num_workers=32)

    logging.basicConfig(level=logging.INFO)
    # Silence tons of `INFO:botocore.credentials:Found credentials in environment variables.` info messages.

    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)

    INPUT_RASTER_BUCKET = config.icechunk.storage_root.strip('s3://')
    INPUT_RASTER_PREFIX = config.icechunk.prefix
    PYRAMID_BUCKET = config.pyramid.storage_root.strip('s3://')
    PYRAMID_PREFIX = config.pyramid.output_prefix
    VAR_LIST = ['rps_2011', 'rps_2047']

    tms = morecantile.tms.get('WebMercatorQuad')
    level = tms.zoom_for_res(30)

    storage = icechunk.s3_storage(
        bucket=INPUT_RASTER_BUCKET, prefix=INPUT_RASTER_PREFIX, from_env=True
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session('main')
    ds = xr.open_zarr(session.store, consolidated=False, chunks='auto')[VAR_LIST]
    ds = ds.proj.assign_crs({'EPSG': 4326})

    pyramid = create_pyramid(
        ds, levels=level, x_dim='longitude', y_dim='latitude', target_shard_bytes=None
    )
    # shards = None while this PR is open: github.com/manzt/zarrita.js/pull/326

    store = from_url(url=f's3://{PYRAMID_BUCKET}/{PYRAMID_PREFIX}', region='us-west-2')
    zstore = ObjectStore(store)
    pyramid.dt.to_zarr(
        zstore, mode='w', encoding=pyramid.encoding, zarr_format=3, align_chunks=False
    )
    zarr.consolidate_metadata(zstore)
