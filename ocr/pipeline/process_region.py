import geopandas as gpd
import xarray as xr
from upath import UPath

from ocr.console import console
from ocr.datasets import catalog
from ocr.risks.fire import calculate_wind_adjusted_risk
from ocr.types import RiskType
from ocr.utils import bbox_tuple_from_xarray_extent, extract_points

from ..config import OCRConfig


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


def calculate_risk(
    config: OCRConfig,
    *,
    region_id: str,
    risk_type: RiskType,
):
    y_slice, x_slice = config.chunking.region_id_to_latlon_slices(region_id=region_id)
    region_geoparquet_uri = config.vector.region_geoparquet_uri

    if risk_type == RiskType.FIRE:
        if config.debug:
            console.log(
                f'Calculating wind risk for region with x_slice: {x_slice} and y_slice: {y_slice}'
            )
        ds = calculate_wind_adjusted_risk(y_slice=y_slice, x_slice=x_slice)
    else:
        raise ValueError(f'Unsupported risk type: {risk_type}')

    config.icechunk.insert_region_uncooperative(
        ds,
        region_id=region_id,
    )

    dset = ds.sel(latitude=y_slice, longitude=x_slice)

    buildings_gdf = sample_risk_to_buildings(ds=dset)

    # Write to geoparquet
    outpath = UPath(f'{region_geoparquet_uri}/{region_id}.parquet')
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True, exist_ok=True)
    buildings_gdf.to_parquet(
        str(outpath),
        index=False,
        compression='zstd',
        geometry_encoding='WKB',
        write_covering_bbox=True,
        schema_version='1.1.0',
    )
    if config.debug:
        console.log(f'Wrote sampled risk data for region {region_id} to {outpath}')
