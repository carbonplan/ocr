# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type r7a.2xlarge
# COILED --tag project=OCR


import duckdb
import geopandas as gpd
import icechunk
import xarray as xr

from ocr.utils import extract_points

# Hardcoding LA region icechunk store, eventually we will use 'template' in catalog
region_id = 'y10_x2'
bucket = 'carbonplan-ocr'
prefix = f'intermediate/fire-risk/tensor/TEST/BP_{region_id}'
storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session('main')

ds = xr.open_zarr(session.store, consolidated=False, chunks={})
ds.rio.write_crs(5070, inplace=True)

# subset gdf from dataset extent
x_min = int(ds.x.min())
x_max = int(ds.x.max())
y_min = int(ds.y.min())
y_max = int(ds.y.max())
bbox = (x_min, y_min, x_max, y_max)

duckdb.sql("""INSTALL SPATIAL; LOAD SPATIAL; INSTALL HTTPFS; LOAD HTTPFS;""")
buildings_table = duckdb.sql(f"""SELECT bbox, bbox_4326, ST_AsText(geometry_4326) as geometry_4326, ST_AsText(geometry) as geometry
    FROM 's3://carbonplan-ocr/input/fire-risk/vector/CONUS_overture_buildings_5070_2025-03-19.1.parquet'
    WHERE
        bbox.xmin BETWEEN {bbox[0]} AND {bbox[2]} AND
        bbox.ymin BETWEEN {bbox[1]} AND {bbox[3]}""").df()

# https://github.com/duckdb/duckdb-spatial/issues/311
buildings_table['geometry'] = gpd.GeoSeries.from_wkt(buildings_table['geometry'])
buildings_table['geometry_4326'] = gpd.GeoSeries.from_wkt(buildings_table['geometry_4326'])

buildings_subset_gdf = gpd.GeoDataFrame(buildings_table, geometry='geometry', crs='EPSG:5070')


buildings_subset_gdf['BP'] = extract_points(buildings_subset_gdf, ds['BP'])

buildings_subset_gdf[['BP', 'bbox_4326', 'geometry_4326', 'geometry']].to_parquet(
    f's3://carbonplan-ocr/intermediate/fire-risk/vector/{region_id}.parquet',
    compression='zstd',
    geometry_encoding='WKB',
    write_covering_bbox=True,
    schema_version='1.1.0',
)
