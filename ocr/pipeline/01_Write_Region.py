# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m7a.large
# COILED --tag project=OCR
from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    pass


def sample_risk_region(region_id: str, branch: str):
    """Samples wind adjusted risk and USFS RPS values from Icechunk store to building polygons based on region_id extent.
    Writes sampled geoparquet to s3 with name based on region_id.

    Args:
        region_id (str): Valid `region_id` defined in chunking_config.py. Ex: y2_x4
    """

    import geopandas as gpd
    import xarray as xr

    from ocr import catalog
    from ocr.chunking_config import ChunkingConfig
    from ocr.template import IcechunkConfig
    from ocr.utils import bbox_tuple_from_xarray_extent, extract_points

    # TODO: We should use logging here!
    print(f'Creating building samples for region: {region_id}')

    # Note: This is still hardcoded to USFS chunking config!
    config = ChunkingConfig()
    icechunk_config = IcechunkConfig(branch=branch)

    icechunk_repo_and_session = icechunk_config.repo_and_session()
    y_slice, x_slice = config.region_id_to_latlon_slices(region_id=region_id)
    ds = xr.open_zarr(
        icechunk_repo_and_session['session'].store, consolidated=False, chunks={}
    ).sel(latitude=y_slice, longitude=x_slice)
    # Build bbox extent from region_id subset.
    bbox = bbox_tuple_from_xarray_extent(ds, x_name='longitude', y_name='latitude')

    building_parquet = catalog.get_dataset('conus-overture-buildings')

    buildings_table = building_parquet.query_geoparquet(
        """SELECT bbox, ST_AsText(geometry) as geometry
        FROM read_parquet('{s3_path}')"""
        + f"""
        WHERE
        bbox.xmin BETWEEN {bbox[0]} AND {bbox[2]} AND
        bbox.ymin BETWEEN {bbox[1]} AND {bbox[3]}"""
    ).df()

    # # https://github.com/duckdb/duckdb-spatial/issues/311
    buildings_table['geometry'] = gpd.GeoSeries.from_wkt(buildings_table['geometry'])
    buildings_subset_gdf = gpd.GeoDataFrame(buildings_table, geometry='geometry', crs='EPSG:4326')

    data_var_list = list(ds.data_vars)
    for var in data_var_list:
        buildings_subset_gdf[var] = extract_points(buildings_subset_gdf, ds[var])

    geom_cols = ['geometry']

    # NOTE / TODO : UPDATE BASED ON VectorConfig()
    buildings_subset_gdf[data_var_list + geom_cols].to_parquet(
        f's3://carbonplan-ocr/intermediate/fire-risk/vector/PIPELINE/{region_id}_2var.parquet',
        compression='zstd',
        geometry_encoding='WKB',
        write_covering_bbox=True,
        schema_version='1.1.0',
    )


def run_wind_region(region_id: str, branch: str):
    """Given a 'region_id', calculate wind adjusted risk and write that region to Icechunk CONUS template.

    Args:
        region_id (str): Valid `region_id` defined in chunking_config.py. Ex: y2_x4
    """

    from distributed import Client

    from ocr.template import insert_region_uncoop
    from ocr.wind import run_wind_adjustment

    client = Client()

    risk_4326_combined = run_wind_adjustment(region_id=region_id)
    # Using the Icechunk uncooperative writes method: https://icechunk.io/en/latest/icechunk-python/parallel/#uncooperative-distributed-writes
    # In this, we are trading performance / more difficult conflict resolution for stateless processing.
    insert_region_uncoop(subset_ds=risk_4326_combined, region_id=region_id, branch=branch)
    # only use dask for xarray, shutdown for duckdb wind sample
    client.shutdown()


@click.command()
@click.option('-r', '--region-id', required=True, help='region_id. ex: y5_x12')
@click.option('-b', '--branch', help='data branch: [QA, prod]. Default QA')
def main(region_id: str, branch: str):
    run_wind_region(region_id, branch)
    # sample_risk_region(region_id, branch)


if __name__ == '__main__':
    main()
