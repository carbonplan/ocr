# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m8g.8xlarge
# COILED --tag project=OCR


# peak 86GB


import duckdb

con = duckdb.connect(database=':memory:')
con.execute("""INSTALL SPATIAL; LOAD SPATIAL; INSTALL HTTPS; LOAD HTTPFS""")

hist_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

con.execute("""
CREATE TEMP TABLE temp_buildings AS
SELECT geometry, risk_2011, risk_2047, wind_risk_2011, wind_risk_2047
FROM read_parquet('s3://carbonplan-ocr/intermediate/fire-risk/vector/prod/consolidated_geoparquet.parquet')
""")

con.execute("""
CREATE TEMP TABLE temp_counties AS
SELECT NAME, geometry
FROM read_parquet('s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/counties.parquet')
""")

con.execute('CREATE INDEX buildings_spatial_idx ON temp_buildings USING RTREE (geometry)')
con.execute('CREATE INDEX counties_spatial_idx ON temp_counties USING RTREE (geometry)')


result = con.query(f"""
COPY (SELECT b.NAME as county_name, count(b.NAME) as building_count,
avg(a.risk_2011) as fire_risk_2011_avg,
avg(a.risk_2047) as fire_risk_2047_avg,
avg(a.wind_risk_2011) as fire_wind_risk_2011_avg,
avg(a.wind_risk_2047) as fire_wind_risk_2047_avg,

histogram(a.risk_2011, {hist_bins}) as risk_2011,
histogram(a.risk_2047, {hist_bins}) as risk_2047,
histogram(a.wind_risk_2011, {hist_bins}) as wind_risk_2011,
histogram(a.wind_risk_2047, {hist_bins}) as wind_risk_2047,

FROM temp_buildings a
JOIN temp_counties b ON ST_Intersects(a.geometry, b.geometry)
GROUP BY b.NAME )
TO 's3://carbonplan-ocr/intermediate/fire-risk/vector/county/county_summary_stats.parquet'
(
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);
""")
