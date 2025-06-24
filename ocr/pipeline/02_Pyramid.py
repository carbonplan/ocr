# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type r7a.2xlarge
# COILED --tag project=OCR
# COILED --name Create_Pyramid
from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    import xarray as xr


# TODO: finish typing
def sel_coarsen(ds: xr.Dataset, factor, dims, **kwargs):
    return ds.sel(**{dim: slice(None, None, factor) for dim in dims})


def create_pyramid(branch: str, reproject: bool = False):
    import xarray as xr
    from ndpyramid import pyramid_create
    from odc.geo.xr import assign_crs

    from ocr.template import IcechunkConfig, PyramidConfig

    pyramid_config = PyramidConfig(branch=branch)
    icechunk_config = IcechunkConfig(branch=branch)

    icechunk_repo_and_session = icechunk_config.repo_and_session(readonly=True)
    ds = xr.open_zarr(icechunk_repo_and_session['session'].store, consolidated=False)
    ds = assign_crs(ds, crs='EPSG:4326')

    if reproject:
        from odc.geo.xr import xr_reproject

        dims = ('y', 'x')
        end_projection = 'EPSG:3857'
        ds = xr_reproject(ds, how=end_projection)
        ds = assign_crs(ds, crs=end_projection)
    else:
        dims = ('latitude', 'longitude')
        end_projection = 'EPSG:4326'
        ds = assign_crs(ds, crs=end_projection)

    # Single level pyramids
    factors = [1]

    pyramid = pyramid_create(
        ds,
        dims=dims,
        factors=factors,
        boundary='trim',
        func=sel_coarsen,
        method_label='slice_coarsen',
        type_label='pick',
    )
    # Note: since datatree doesn't support writing to regions, we default to wiping on overwrite.
    pyramid.to_zarr(pyramid_config.uri, mode='w', zarr_format=3, consolidated=False)


@click.command()
@click.option('-b', '--branch', default='QA', help='data branch: [QA, prod]. Default QA')
@click.option(
    '-r', '--reproject', is_flag=True, help='if True, re-project to EPSG:3857 - web mercator'
)
def main(branch: str, reproject: bool = False):
    create_pyramid(branch, reproject)


if __name__ == '__main__':
    main()
