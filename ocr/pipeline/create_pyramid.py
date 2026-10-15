from ocr.config import OCRConfig


def create_pyramid(config: OCRConfig):
    """Write multiscale overview levels 1..N next to the 30m base (group '0') in the icechunk store."""
    import logging

    import xarray as xr
    import xproj  # noqa
    import zarr
    from topozarr import create_pyramid

    zarr.config.set({'async.concurrency': 128})

    logging.basicConfig(level=logging.INFO)
    # Silence tons of `INFO:botocore.credentials:Found credentials in environment variables.` info messages.
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)

    session = config.icechunk.repo_and_session()['session']
    ds = xr.open_zarr(session.store, group='0', chunks=None, consolidated=False)
    ds = ds.proj.assign_crs(spatial_ref='EPSG:4326')

    pyramid = create_pyramid(ds, levels='auto', x_dim='longitude', y_dim='latitude')
    pyramid.write(session.store, mode='a', levels=list(range(1, pyramid.levels)), progress=True)
    session.commit('add multiscale overviews')
