# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type r7a.xlarge
# COILED --tag project=OCR


import click


def sample_risk_region(region_id: str):
    # reads 'template', does some duckdb/geoparquet reproj, does a .sel, writes to geoparquet
    import geopandas as gpd
    import xarray as xr

    from ocr import catalog
    from ocr.chunking_config import ChunkingConfig
    from ocr.config import TemplateConfig
    from ocr.utils import extract_points, x_y_bbox_tuple_from_xarray_extent

    # TODO: We should use logging here!
    print(f'Creating building samples for region: {region_id}')

    # Note / Warning: This is still hardcoded to USFS chunking config!
    config = ChunkingConfig()

    template_config = TemplateConfig()
    # TODO: Make sure to add a 'readonly' options
    icechunk_config = template_config.init_icechunk_repo(readonly=True)

    ds = xr.open_zarr(icechunk_config['session'].store, consolidated=False, chunks={})

    # tmp, this should be in 01
    ds.rio.write_crs(5070, inplace=True)

    # Note: get a bbox subset of dataset extent - only in full extent template! For now we are circumventing that!
    bbox = x_y_bbox_tuple_from_xarray_extent(ds)

    building_parquet_5070 = catalog.get_dataset('conus-overture-buildings-5070')

    buildings_table = building_parquet_5070.query_geoparquet(
        """SELECT bbox, bbox_4326, ST_AsText(geometry_4326) as geometry_4326, ST_AsText(geometry) as geometry
        FROM read_parquet('{s3_path}')"""
        + f"""
        WHERE
        bbox.xmin BETWEEN {bbox[0]} AND {bbox[2]} AND
        bbox.ymin BETWEEN {bbox[1]} AND {bbox[3]}"""
    ).df()

    # # https://github.com/duckdb/duckdb-spatial/issues/311
    buildings_table['geometry'] = gpd.GeoSeries.from_wkt(buildings_table['geometry'])
    buildings_table['geometry_4326'] = gpd.GeoSeries.from_wkt(buildings_table['geometry_4326'])
    buildings_subset_gdf = gpd.GeoDataFrame(buildings_table, geometry='geometry', crs='EPSG:5070')

    data_var_list = list(ds.data_vars)
    for var in data_var_list:
        buildings_subset_gdf[var] = extract_points(buildings_subset_gdf, ds[var])

    # NOTE: TODO: PARQUET PATH IS HARDCODED
    geom_cols =['bbox_4326', 'geometry_4326', 'geometry']

    buildings_subset_gdf[data_var_list + geom_cols].to_parquet(
        f's3://carbonplan-ocr/intermediate/fire-risk/vector/PIPELINE/{region_id}_2var.parquet',
        compression='zstd',
        geometry_encoding='WKB',
        write_covering_bbox=True,
        schema_version='1.1.0',
    )


def run_wind_region(region_id: str):
    # TODO: / WARNING The chunking config of the end result may differ from the USFS one. We need to make the ChunkingConfig more general most likely
    from ocr import catalog
    from ocr.chunking_config import ChunkingConfig
    from ocr.config import TemplateConfig

    config = ChunkingConfig()
    template_config = TemplateConfig()
    icechunk_config = template_config.init_icechunk_repo()

    # TODO: We should use logging here!
    print(f'Writing wind: processing region: {region_id}')

    y_slice, x_slice = config.region_id_slice_lookup(region_id=region_id)

    # TEMPORARY! This can be replaced the a contained 'wind' function
    ds = catalog.get_dataset('USFS-wildfire-risk-communities').to_xarray()[['BP']]
    ds['BP'] = ds['BP'].astype('float32')
    import random
    ds['BP_wind_adjusted'] = ds['BP'] + random.uniform(0.0, 0.01)

    subset_ds = ds.isel(y=y_slice, x=x_slice)

    # TEMPORARY! We will want to write to a template
    # for now, call icechunk.config to init
    subset_ds.rio.write_crs(5070, inplace=True)

    subset_ds.to_zarr(
        icechunk_config['session'].store,
        mode='w',
        consolidated=False,
    )

    icechunk_config['session'].commit(f'{region_id}')
    # TODO: When doing uncoop writes, we will have to update this all!
    # eventually this should be region_id?


@click.command()
@click.option('-r', '--region-id', required=True, help='region_id. ex: y5_x12')
def main(region_id: str):
    """We need a wrapper function like this to pass in CLI args, but this can call the wind process script.. I think"""
    run_wind_region(region_id)
    sample_risk_region(region_id)


if __name__ == '__main__':
    main()

# uv run python main.py -r y9_x1 -c
# uv run python main.py -r y9_x2 -c
# uv run python main.py -r y10_x2 -c
# uv run python main.py -r y10_x3 -c
