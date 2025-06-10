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
    from ocr.config import TemplateConfig
    from ocr.utils import bbox_tuple_from_xarray_extent, extract_points

    # TODO: We should use logging here!
    print(f'Creating building samples for region: {region_id}')

    # Note / Warning: This is still hardcoded to USFS chunking config!
    # config = ChunkingConfig()

    template_config = TemplateConfig()
    icechunk_config = template_config.init_icechunk_repo(readonly=True)

    ds = xr.open_zarr(icechunk_config['session'].store, consolidated=False, chunks={})

    # Note: get a bbox subset of dataset extent - only in full extent template! For now we are circumventing that!
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

    # NOTE: TODO: PARQUET PATH IS HARDCODED
    buildings_subset_gdf[data_var_list + geom_cols].to_parquet(
        f's3://carbonplan-ocr/intermediate/fire-risk/vector/PIPELINE/{region_id}_2var.parquet',
        compression='zstd',
        geometry_encoding='WKB',
        write_covering_bbox=True,
        schema_version='1.1.0',
    )


def run_wind_region(region_id: str):
    from odc.geo.xr import assign_crs
    from rasterio.warp import Resampling

    from ocr import catalog
    from ocr.chunking_config import ChunkingConfig
    from ocr.config import TemplateConfig
    from ocr.utils import (
        lon_to_180,
    )
    from ocr.wind import (
        apply_mode_calc,
        apply_wind_directional_convolution,
        classify_wind_directions,
        create_composite_bp_map,
    )

    config = ChunkingConfig()

    # init icechunk repo
    template_config = TemplateConfig()
    # should we add the EPSG / spatial ref here in the template creation?
    icechunk_config = template_config.init_icechunk_repo()

    climate_run = catalog.get_dataset('2011-climate-run-30m-4326').to_xarray()[
        ['BP']
    ]  # .to_dataset().isel(latitude=slice(60000,66000), longitude=slice(100000,104500))
    rps_30 = catalog.get_dataset('USFS-wildfire-risk-communities-4326').to_xarray()[
        ['BP', 'CRPS', 'RPS']
    ]  # .isel(y=y_slice, x=x_slice)
    important_days = catalog.get_dataset('era5-fire-weather-days').to_xarray()[
        ['sfcWindfromdir']
    ]  # .isel(y=y_slice, x=x_slice)
    important_days = lon_to_180(important_days)

    y_slice, x_slice = config.region_id_to_latlon_slices(region_id=region_id)
    rps_30_subset = rps_30.sel(latitude=y_slice, longitude=x_slice)
    climate_run_subset = climate_run.sel(latitude=y_slice, longitude=x_slice)
    wind_directions = important_days.sel(latitude=y_slice, longitude=x_slice)

    blurred_bp = apply_wind_directional_convolution(rps_30_subset['BP'], iterations=3)
    direction_indices = classify_wind_directions(wind_directions).chunk(dict(time=-1))
    direction_modes = apply_mode_calc(direction_indices).compute()

    direction_modes_sfc = assign_crs(direction_modes['sfcWindfromdir'], crs='EPSG:4326')
    blurred_bp = assign_crs(blurred_bp, crs='EPSG:4326')
    # TODO! We can probably use xarray's interp_like, instead of reproject match, since we have the same coords
    # direction_modes_sfc.interp_like(blurred_bp,method="nearest").plot()
    wind_direction_reprojected = direction_modes_sfc.rio.reproject_match(
        blurred_bp, resampling=Resampling.nearest
    ).rename({'y': 'latitude', 'x': 'longitude'})

    wind_informed_bp = create_composite_bp_map(blurred_bp, wind_direction_reprojected).drop_vars(
        'direction'
    )

    risk_4326 = (wind_informed_bp * rps_30_subset['CRPS']).to_dataset(name='wind_risk')
    # add in USFS 30m 4326 risk score (burn probability)
    risk_4326['USFS_RPS'] = rps_30_subset['RPS']

    # assign crs and reproject EPSG:5070
    # risk_4326 = assign_crs(risk_4326, crs='EPSG:4326')
    # risk_5070 = xr_reproject(risk_4326, how='EPSG:5070')
    import ipdb

    ipdb.set_trace()
    risk_4326.to_zarr(
        icechunk_config['session'].store,
        mode='w',
        consolidated=False,
    )
    # ValueError: failed to prevent overwriting existing key grid_mapping in attrs. This is probably an encoding field used by xarray to describe how a variable is serialized. To proceed, remove this key from the variable's attributes manually.
    icechunk_config['session'].commit(f'{region_id}')


@click.command()
@click.option('-r', '--region-id', required=True, help='region_id. ex: y5_x12')
def main(region_id: str):
    """We need a wrapper function like this to pass in CLI args, but this can call the wind process script.. I think"""
    print(region_id)
    # run_wind_region(region_id)
    sample_risk_region(region_id)


if __name__ == '__main__':
    main()
