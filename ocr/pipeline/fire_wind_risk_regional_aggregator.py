# peak 96GB!

from ocr.types import Branch
from ocr.console import console

def compute_regional_fire_wind_risk_statistics(branch: Branch):
    import duckdb

    
    s3_base = "s3://carbonplan-ocr"
    input_base = f"{s3_base}/input/fire-risk/vector"
    intermediate_base = f"{s3_base}/intermediate/fire-risk/vector/{branch.value}"
    
  
    consolidated_buildings_path = f"{intermediate_base}/consolidated_geoparquet.parquet"
    counties_path = f"{input_base}/aggregated_regions/counties.parquet"
    tracts_path = f"{input_base}/aggregated_regions/tracts/tracts.parquet"
    
    
    county_output_path = f"{intermediate_base}/region_aggregation/county/county_summary_stats.parquet"
    tract_output_path = f"{intermediate_base}/region_aggregation/tract/tract_summary_stats.parquet"


    con = duckdb.connect(database=':memory:')
    con.execute("""INSTALL SPATIAL; LOAD SPATIAL; INSTALL HTTPS; LOAD HTTPFS""")

    hist_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    console.log(f'Computing regional fire and wind risk statistics for branch: {branch.value} with histogram bins: {hist_bins}')
    console.log(f'Using consolidated buildings path: {consolidated_buildings_path}')

    # tmp table for buildings
    con.execute(f"""
        CREATE TEMP TABLE buildings AS
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


        FROM read_parquet('{consolidated_buildings_path}')
        """)


    console.log('Temporary tables created for buildings, counties, and tracts.')
    console.log(f'Counties path: {counties_path}')
    # tmp table for geoms
    con.execute("""
        CREATE TEMP TABLE county AS
        SELECT NAME, geometry
        FROM read_parquet('{counties_path}')
        """)
    
    console.log('Temporary table created for counties.')
    console.log(f'Tracts path: {tracts_path}')
    # tmp table for tracts
    con.execute("""
        CREATE TEMP TABLE temp_tracts AS
        SELECT GEOID, geometry
        FROM read_parquet('{tracts_path}')
        """)

    # create spatial index on geom cols
    con.execute('CREATE INDEX buildings_spatial_idx ON temp_buildings USING RTREE (geometry)')
    con.execute('CREATE INDEX counties_spatial_idx ON temp_counties USING RTREE (geometry)')
    con.execute('CREATE INDEX tracts_spatial_idx ON temp_tracts USING RTREE (geometry)')

    # county level histograms
    console.log('Calculating county level statistics...')
    console.log(f'County output path: {county_output_path}')
    con.query(f"""COPY (SELECT b.NAME as county_name, count(a.risk_2011_horizon_1) as building_count,
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

        map_values(histogram(CASE WHEN a.risk_2011_horizon_1 <> 0 AND a.risk_2011_horizon_1 <= 100 THEN a.risk_2011_horizon_1 END, {hist_bins})) as nonzero_hist_risk_2011_horizon_1,
        map_values(histogram(CASE WHEN a.risk_2011_horizon_15 <> 0 AND a.risk_2011_horizon_15 <= 100 THEN a.risk_2011_horizon_15 END, {hist_bins})) as nonzero_hist_risk_2011_horizon_15,
        map_values(histogram(CASE WHEN a.risk_2011_horizon_30 <> 0 AND a.risk_2011_horizon_30 <= 100 THEN a.risk_2011_horizon_30 END, {hist_bins})) as nonzero_hist_risk_2011_horizon_30,

        map_values(histogram(CASE WHEN a.risk_2047_horizon_1 <> 0 AND a.risk_2047_horizon_1 <= 100 THEN a.risk_2047_horizon_1 END, {hist_bins})) as nonzero_hist_risk_2047_horizon_1,
        map_values(histogram(CASE WHEN a.risk_2047_horizon_15 <> 0 AND a.risk_2047_horizon_15 <= 100 THEN a.risk_2047_horizon_15 END, {hist_bins})) as nonzero_hist_risk_2047_horizon_15,
        map_values(histogram(CASE WHEN a.risk_2047_horizon_30 <> 0 AND a.risk_2047_horizon_30 <= 100 THEN a.risk_2047_horizon_30 END, {hist_bins})) as nonzero_hist_risk_2047_horizon_30,

        map_values(histogram(CASE WHEN a.wind_risk_2011_horizon_1 <> 0 AND a.wind_risk_2011_horizon_1 <= 100 THEN a.wind_risk_2011_horizon_1 END, {hist_bins})) as nonzero_hist_wind_risk_2011_horizon_1,
        map_values(histogram(CASE WHEN a.wind_risk_2011_horizon_15 <> 0 AND a.wind_risk_2011_horizon_15 <= 100 THEN a.wind_risk_2011_horizon_15 END, {hist_bins})) as nonzero_hist_wind_risk_2011_horizon_15,
        map_values(histogram(CASE WHEN a.wind_risk_2011_horizon_30 <> 0 AND a.wind_risk_2011_horizon_30 <= 100 THEN a.wind_risk_2011_horizon_30 END, {hist_bins})) as nonzero_hist_wind_risk_2011_horizon_30,

        map_values(histogram(CASE WHEN a.wind_risk_2047_horizon_1 <> 0 AND a.wind_risk_2047_horizon_1 <= 100 THEN a.wind_risk_2047_horizon_1 END, {hist_bins})) as nonzero_hist_wind_risk_2047_horizon_1,
        map_values(histogram(CASE WHEN a.wind_risk_2047_horizon_15 <> 0 AND a.wind_risk_2047_horizon_15 <= 100 THEN a.wind_risk_2047_horizon_15 END, {hist_bins})) as nonzero_hist_wind_risk_2047_horizon_15,
        map_values(histogram(CASE WHEN a.wind_risk_2047_horizon_30 <> 0 AND a.wind_risk_2047_horizon_30 <= 100 THEN a.wind_risk_2047_horizon_30 END, {hist_bins})) as nonzero_hist_wind_risk_2047_horizon_30,

        b.geometry as geometry
        FROM temp_buildings a
        JOIN temp_counties b ON ST_Intersects(a.geometry, b.geometry)
        GROUP BY b.NAME, b.geometry )
        TO '{county_output_path}'
        (
                FORMAT 'parquet',
                COMPRESSION 'zstd',
                OVERWRITE_OR_IGNORE true);
        """)

    # tract level histograms - we should refactor this mostly shared SQL
    console.log('Calculating tract level statistics...')
    console.log(f'Tract output path: {tract_output_path}')
    con.query(f"""COPY (SELECT b.GEOID as tract_geoid, count(a.risk_2011_horizon_1) as building_count,
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
        JOIN temp_tracts b ON ST_Intersects(a.geometry, b.geometry)
        GROUP BY b.GEOID, b.geometry )
        TO '{tract_output_path}'
        (
                FORMAT 'parquet',
                COMPRESSION 'zstd',
                OVERWRITE_OR_IGNORE true);
    """
    con.execute(merge_and_write)

    console.log('Regional fire and wind risk statistics computed and written to:')
    console.log(f'County statistics written to: {county_output_path}')
    console.log(f'Tract statistics written to: {tract_output_path}')    


