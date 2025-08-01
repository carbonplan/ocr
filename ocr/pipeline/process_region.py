import geopandas as gpd
import icechunk
import xarray as xr

from ocr.config import OCRConfig
from ocr.console import console
from ocr.datasets import catalog
from ocr.icechunk_utils import insert_region_uncoop, region_id_exists_in_repo
from ocr.risks.fire import calculate_wind_adjusted_risk
from ocr.types import RiskType
from ocr.utils import bbox_tuple_from_xarray_extent, extract_points


def write_region_to_icechunk(session: icechunk.Session, *, ds: xr.Dataset, region_id: str):
    insert_region_uncoop(
        session=session,
        subset_ds=ds,
        region_id=region_id,
    )


def sample_risk_to_buildings(*, ds: xr.Dataset) -> gpd.GeoDataFrame:
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

    return buildings_gdf


def calculate_risk(config: OCRConfig, *, region_id: str, risk_type: RiskType):
    if region_id not in config.chunking.valid_region_ids:
        raise ValueError(
            f'{region_id} is an invalid region_id. Valid IDs include: {config.chunking.valid_region_ids}'
        )

    # Check if region already exists
    if region_id_exists_in_repo(config.icechunk.repo_and_session()['repo'], region_id=region_id):
        raise ValueError(
            f'Region {region_id} already exists in Icechunk store.'
            'Please provide a new region_id or use the wipe flag to overwrite existing data.'
        )
    y_slice, x_slice = config.chunking.region_id_to_latlon_slices(region_id=region_id)
    if risk_type == RiskType.FIRE:
        ds = calculate_wind_adjusted_risk(y_slice=y_slice, x_slice=x_slice)
    else:
        raise ValueError(f'Unsupported risk type: {risk_type}')

    write_region_to_icechunk(
        session=config.icechunk.repo_and_session()['session'],
        ds=ds,
        region_id=region_id,
    )

    dset = ds.sel(latitude=y_slice, longitude=x_slice)

    buildings_gdf = sample_risk_to_buildings(ds=dset)

    # Write to geoparquet
    outpath = f'{config.vector.region_geoparquet_uri}{region_id}.parquet'
    buildings_gdf.to_parquet(
        outpath,
        compression='zstd',
        geometry_encoding='WKB',
        write_covering_bbox=True,
        schema_version='1.1.0',
    )
    console.log(f'Wrote sampled risk data for region {region_id} to {outpath}')
