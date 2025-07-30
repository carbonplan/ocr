import geopandas as gpd
import xarray as xr

from ocr.chunking_config import ChunkingConfig
from ocr.datasets import catalog
from ocr.template import (
    IcechunkConfig,
    VectorConfig,
    insert_region_uncoop,
    region_id_exists_in_repo,
)
from ocr.types import Branch, RiskType
from ocr.utils import bbox_tuple_from_xarray_extent, extract_points
from ocr.wind import calculate_wind_adjusted_risk


def write_region_to_icechunk(ds: xr.Dataset, region_id: str, branch: Branch, wipe: bool):
    insert_region_uncoop(
        subset_ds=ds,
        region_id=region_id,
        branch=branch,
        wipe=wipe,  # Wipe is handled at the IcechunkConfig level
    )


def sample_risk_to_buildings(region_id: str, branch: Branch):
    config = ChunkingConfig()
    icechunk_config = IcechunkConfig(branch=branch.value)
    vector_config = VectorConfig(branch=branch.value)

    icechunk_repo_and_session = icechunk_config.repo_and_session(readonly=True)
    y_slice, x_slice = config.region_id_to_latlon_slices(region_id=region_id)
    ds = xr.open_dataset(
        icechunk_repo_and_session['session'].store, engine='zarr', consolidated=False, chunks={}
    ).sel(latitude=y_slice, longitude=x_slice)
    # Create bounding box from region
    bbox = bbox_tuple_from_xarray_extent(ds, x_name='longitude', y_name='latitude')
    # Query buildings within the region
    building_parquet = catalog.get_dataset('conus-overture-buildings')
    buildings_table = building_parquet.query_geoparquet(
        """SELECT bbox, ST_AsText(geometry) as geometry
        FROM read_parquet('{s3_path}')"""
        + f"""
        WHERE
        bbox.xmin BETWEEN {bbox[0]} AND {bbox[2]} AND
        bbox.ymin BETWEEN {bbox[1]} AND {bbox[3]}"""
    ).df()

    # Convert to GeoDataFrame
    buildings_table['geometry'] = gpd.GeoSeries.from_wkt(buildings_table['geometry'])
    buildings_gdf = gpd.GeoDataFrame(buildings_table, geometry='geometry', crs='EPSG:4326')

    # Sample risk values at building locations
    data_var_list = list(ds.data_vars)
    for var in data_var_list:
        buildings_gdf[var] = extract_points(buildings_gdf, ds[var])

    # Remove any buildings with NaN values (outside CONUS)
    geom_cols = ['geometry']
    buildings_gdf = buildings_gdf[data_var_list + geom_cols].dropna(subset=data_var_list)

    # Write to geoparquet
    outpath = f'{vector_config.region_geoparquet_uri}{region_id}.parquet'
    buildings_gdf.to_parquet(
        outpath,
        compression='zstd',
        geometry_encoding='WKB',
        write_covering_bbox=True,
        schema_version='1.1.0',
    )


def calculate_risk(region_id: str, risk_type: RiskType, branch: Branch, wipe: bool):
    config = ChunkingConfig()
    if region_id not in config.valid_region_ids:
        raise ValueError(
            f'{region_id} is an invalid region_id. Valid IDs include: {config.valid_region_ids}'
        )

    # Check if region already exists
    if region_id_exists_in_repo(region_id=region_id, branch=branch.value):
        raise ValueError(
            f'Region {region_id} already exists in Icechunk store.'
            f' {branch.value} branch. Please provide a new region_id or use the wipe flag to overwrite existing data.'
        )
    if risk_type == RiskType.WIND:
        ds = calculate_wind_adjusted_risk(region_id=region_id)
    else:
        raise ValueError(f'Unsupported risk type: {risk_type}')

    write_region_to_icechunk(ds=ds, region_id=region_id, branch=branch, wipe=wipe)
    sample_risk_to_buildings(region_id=region_id, branch=branch)
