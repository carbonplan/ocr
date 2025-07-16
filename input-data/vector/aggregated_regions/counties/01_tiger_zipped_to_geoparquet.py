import geopandas as gpd

gdf = gpd.read_file(
    'https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip',
    columns=['NAME', 'geometry'],
)

gdf.to_parquet(
    's3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/counties.parquet',
    compression='zstd',
    geometry_encoding='WKB',
    write_covering_bbox=True,
    schema_version='1.1.0',
)
