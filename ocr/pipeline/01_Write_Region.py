# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m7i.xlarge
# COILED --tag project=OCR
from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    import xarray as xr


def sample_risk_region(region_id: str):
    """Samples wind adjusted risk and USFS RPS values from Icechunk store to building polygons based on region_id extent.
    Writes sampled geoparquet to s3 with name based on region_id.

    Args:
        region_id (str): Valid `region_id` defined in chunking_config.py. Ex: y2_x4
    """

    import geopandas as gpd
    import xarray as xr

    from ocr import catalog
    from ocr.chunking_config import ChunkingConfig
    from ocr.template import TemplateConfig
    from ocr.utils import bbox_tuple_from_xarray_extent, extract_points

    # TODO: We should use logging here!
    print(f'Creating building samples for region: {region_id}')

    # Note: This is still hardcoded to USFS chunking config!
    config = ChunkingConfig()
    template_config = TemplateConfig()

    icechunk_repo_and_session = template_config.repo_and_session()
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

    # NOTE: PARQUET PATH IS HARDCODED
    buildings_subset_gdf[data_var_list + geom_cols].to_parquet(
        f's3://carbonplan-ocr/intermediate/fire-risk/vector/PIPELINE/{region_id}_2var.parquet',
        compression='zstd',
        geometry_encoding='WKB',
        write_covering_bbox=True,
        schema_version='1.1.0',
    )


def classify_wind(
    climate_run_subset: xr.Dataset, direction_modes_sfc: xr.Dataset, rps_30_subset: xr.Dataset
) -> xr.Dataset:
    from odc.geo.xr import assign_crs

    from ocr.wind import (
        apply_wind_directional_convolution,
        create_composite_bp_map,
    )

    # Build and apply wind adjustment
    blurred_bp = apply_wind_directional_convolution(climate_run_subset['BP'], iterations=3)

    blurred_bp = assign_crs(blurred_bp, crs='EPSG:4326')
    # Switched to xarray interp_like to since both datasets have matching EPSG codes.
    # wind_direction_reprojected = direction_modes_sfc.rio.reproject_match(
    #     blurred_bp, resampling=Resampling.nearest
    # ).rename({'y': 'latitude', 'x': 'longitude'})
    # wind_direction_reprojected = assign_crs(wind_direction_reprojected, crs='EPSG:4326')
    # import xarray.testing as xrt
    # xrt.assert_equal(wind_direction_reprojected, wind_direction_reprojected_xr)

    # Adding int coercion because rasterio outputs ints, while scipy/xarray outputs floats.
    wind_direction_reprojected = direction_modes_sfc.interp_like(
        blurred_bp, method='nearest'
    ).astype(int)

    wind_informed_bp = create_composite_bp_map(blurred_bp, wind_direction_reprojected).drop_vars(
        'direction'
    )
    # Fix tiny FP misalignment in .sel of lat/lon between two datasets.
    wind_informed_bp_float_corrected = wind_informed_bp.assign_coords(
        latitude=rps_30_subset.latitude, longitude=rps_30_subset.longitude
    )
    return wind_informed_bp_float_corrected


def run_wind_region(region_id: str):
    """Given a 'region_id', calculate wind adjusted risk and write that region to Icechunk CONUS template.

    Args:
        region_id (str): Valid `region_id` defined in chunking_config.py. Ex: y2_x4
    """

    from odc.geo.xr import assign_crs

    from ocr import catalog
    from ocr.chunking_config import ChunkingConfig
    from ocr.template import insert_region_uncoop
    from ocr.utils import (
        lon_to_180,
    )
    from ocr.wind import (
        apply_mode_calc,
        classify_wind_directions,
    )

    config = ChunkingConfig()

    # Open input dataset: USFS 30m community risk, USFS 30m interpolated 2011 climate runs and 1/4 degree? ERA5 Wind.
    climate_run_2011 = catalog.get_dataset('2011-climate-run-30m-4326').to_xarray()[['BP']]
    climate_run_2047 = catalog.get_dataset('2047-climate-run-30m-4326').to_xarray()[['BP']]

    rps_30 = catalog.get_dataset('USFS-wildfire-risk-communities-4326').to_xarray()[
        ['BP', 'CRPS', 'RPS']
    ]
    important_days = catalog.get_dataset('era5-fire-weather-days').to_xarray()[['sfcWindfromdir']]
    # TODO: Input datasets should already be pre-processed, so this transform should be done upstream.
    important_days = lon_to_180(important_days)

    y_slice, x_slice = config.region_id_to_latlon_slices(region_id=region_id)
    rps_30_subset = rps_30.sel(latitude=y_slice, longitude=x_slice)
    climate_run_2011_subset = climate_run_2011.sel(latitude=y_slice, longitude=x_slice)
    climate_run_2047_subset = climate_run_2047.sel(latitude=y_slice, longitude=x_slice)

    # Since important_days / wind is a lower resolution (.25 degrees?), we add in spatial buffer to match the resolution.
    wind_res = 0.25
    buffer = wind_res * 2  # add in a 2x buffer of the resolution
    buffered_y_slice = slice(y_slice.start + buffer, y_slice.stop - buffer, y_slice.step)
    buffered_x_slice = slice(x_slice.start - buffer, x_slice.stop + buffer, x_slice.step)

    wind_directions = important_days.sel(latitude=buffered_y_slice, longitude=buffered_x_slice)
    direction_indices = classify_wind_directions(wind_directions).chunk(dict(time=-1))
    direction_modes = apply_mode_calc(direction_indices).compute()

    direction_modes_sfc = assign_crs(direction_modes['sfcWindfromdir'], crs='EPSG:4326')

    wind_informed_bp_float_corrected_2011 = classify_wind(
        climate_run_subset=climate_run_2011_subset,
        direction_modes_sfc=direction_modes_sfc,
        rps_30_subset=rps_30_subset,
    )
    wind_informed_bp_float_corrected_2047 = classify_wind(
        climate_run_subset=climate_run_2047_subset,
        direction_modes_sfc=direction_modes_sfc,
        rps_30_subset=rps_30_subset,
    )

    # Add in USFS 30m 4326 RPS (Risk to Potential Structures) for QA comparison
    risk_4326_combined = rps_30_subset['RPS'].to_dataset(name='USFS_RPS')

    # Adjust USFS 30m CRPS with wind informed burn probability
    risk_4326_combined['wind_risk_2011'] = (
        wind_informed_bp_float_corrected_2011 * rps_30_subset['CRPS']
    )
    risk_4326_combined['wind_risk_2047'] = (
        wind_informed_bp_float_corrected_2047 * rps_30_subset['CRPS']
    )

    risk_4326_combined = risk_4326_combined.drop_vars(['spatial_ref'])
    # Using the Icechunk uncooperative writes method: https://icechunk.io/en/latest/icechunk-python/parallel/#uncooperative-distributed-writes
    # In this, we are trading performance / more difficult conflict resolution for stateless processing.
    insert_region_uncoop(subset_ds=risk_4326_combined, region_id=region_id)


@click.command()
@click.option('-r', '--region-id', required=True, help='region_id. ex: y5_x12')
def main(region_id: str):
    run_wind_region(region_id)
    sample_risk_region(region_id)


if __name__ == '__main__':
    main()
