# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m8g.8xlarge
# COILED --tag project=OCR

# peak 96GB!


import duckdb

con = duckdb.connect(database=':memory:')
con.execute("""INSTALL SPATIAL; LOAD SPATIAL; INSTALL HTTPS; LOAD HTTPFS""")

hist_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


con.execute("""
CREATE TEMP TABLE temp_buildings AS
SELECT geometry,


round(risk_2011 * 100.0, 2) as risk_2011_horizon_1,
round((1.0 - POWER((1 - risk_2011), 15)) * 100.0, 2) as risk_2011_horizon_15,
round((1.0 - POWER((1 - risk_2011), 30)) * 100.0, 2) as risk_2011_horizon_30,

round(risk_2047 * 100.0, 2) as risk_2047_horizon_1,
round((1.0 - POWER((1 - risk_2047), 15)) * 100.0, 2) as risk_2047_horizon_15,
round((1.0 - POWER((1 - risk_2047), 30)) * 100.0, 2) as risk_2047_horizon_30,

round(wind_risk_2011 * 100.0, 2) as wind_risk_2011_horizon_1,
round((1.0 - POWER((1 - wind_risk_2011), 15)) * 100.0, 2) as wind_risk_2011_horizon_15,
round((1.0 - POWER((1 - wind_risk_2011), 30)) * 100.0, 2) as wind_risk_2011_horizon_30,

round(wind_risk_2047 * 100.0, 2) as wind_risk_2047_horizon_1,
round((1.0 - POWER((1 - wind_risk_2047), 15)) * 100.0, 2) as wind_risk_2047_horizon_15,
round((1.0 - POWER((1 - wind_risk_2047), 30)) * 100.0, 2) as wind_risk_2047_horizon_30,


FROM read_parquet('s3://carbonplan-ocr/intermediate/fire-risk/vector/prod/consolidated_geoparquet.parquet')
""")


con.execute("""
CREATE TEMP TABLE temp_counties AS
SELECT NAME, geometry
FROM read_parquet('s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/counties.parquet')
""")

con.execute('CREATE INDEX buildings_spatial_idx ON temp_buildings USING RTREE (geometry)')
con.execute('CREATE INDEX counties_spatial_idx ON temp_counties USING RTREE (geometry)')


result = con.query(f"""COPY (SELECT b.NAME as county_name, count(b.NAME) as building_count,
round(avg(a.risk_2011_horizon_1), 2) as avg_risk_2011_horizon_1,
round(avg(a.risk_2011_horizon_15), 2) as avg_risk_2011_horizon_15,
round(avg(a.risk_2011_horizon_30), 2) as avg_risk_2011_horizon_30,
round(avg(a.risk_2047_horizon_1), 2) as avg_risk_2047_horizon_1,
round(avg(a.risk_2047_horizon_15), 2) as avg_risk_2047_horizon_15,
round(avg(a.risk_2047_horizon_30), 2) as avg_risk_2047_horizon_30,
round(avg(a.wind_risk_2011_horizon_1), 2) as avg_wind_risk_2011_horizon_1,
round(avg(a.wind_risk_2011_horizon_15), 2) as avg_wind_risk_2011_horizon_15,
round(avg(a.wind_risk_2011_horizon_30), 2) as avg_wind_risk_2011_horizon_30,
round(avg(a.wind_risk_2047_horizon_1), 2) as avg_wind_risk_2047_horizon_1,
round(avg(a.wind_risk_2047_horizon_15), 2) as avg_wind_risk_2047_horizon_15,
round(avg(a.wind_risk_2047_horizon_30), 2) as avg_wind_risk_2047_horizon_30,
map_values(histogram(a.risk_2011_horizon_1, {hist_bins})) as risk_2011_horizon_1,
map_values(histogram(a.risk_2011_horizon_15, {hist_bins})) as risk_2011_horizon_15,
map_values(histogram(a.risk_2011_horizon_30, {hist_bins})) as risk_2011_horizon_30,
map_values(histogram(a.risk_2047_horizon_1, {hist_bins})) as risk_2047_horizon_1,
map_values(histogram(a.risk_2047_horizon_15, {hist_bins})) as risk_2047_horizon_15,
map_values(histogram(a.risk_2047_horizon_30, {hist_bins})) as risk_2047_horizon_30,
map_values(histogram(a.wind_risk_2011_horizon_1, {hist_bins})) as wind_risk_2011_horizon_1,
map_values(histogram(a.wind_risk_2011_horizon_15, {hist_bins})) as wind_risk_2011_horizon_15,
map_values(histogram(a.wind_risk_2011_horizon_30, {hist_bins})) as wind_risk_2011_horizon_30,
map_values(histogram(a.wind_risk_2047_horizon_1, {hist_bins})) as wind_risk_2047_horizon_1,
map_values(histogram(a.wind_risk_2047_horizon_15, {hist_bins})) as wind_risk_2047_horizon_15,
map_values(histogram(a.wind_risk_2047_horizon_30, {hist_bins})) as wind_risk_2047_horizon_30,
b.geometry as geometry
FROM temp_buildings a
JOIN temp_counties b ON ST_Intersects(a.geometry, b.geometry)
GROUP BY b.NAME, b.geometry )
TO 's3://carbonplan-ocr/intermediate/fire-risk/vector/county/county_summary_stats.parquet'
(
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);
""")
