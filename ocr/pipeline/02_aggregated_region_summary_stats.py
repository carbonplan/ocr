# peak 96GB!

import click


def create_summary_stats(branch: str):
    import duckdb

    con = duckdb.connect(database=':memory:')
    con.execute("""INSTALL SPATIAL; LOAD SPATIAL; INSTALL HTTPS; LOAD HTTPFS""")

    hist_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # tmp table for buildings
    con.execute(f"""
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


        FROM read_parquet('s3://carbonplan-ocr/intermediate/fire-risk/vector/{branch}/consolidated_geoparquet.parquet')
        """)

    # tmp table for geoms
    con.execute("""
        CREATE TEMP TABLE temp_counties AS
        SELECT NAME, geometry
        FROM read_parquet('s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/counties.parquet')
        """)

    # tmp table for tracts
    con.execute("""
        CREATE TEMP TABLE temp_tracts AS
        SELECT GEOID, geometry
        FROM read_parquet('s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/tracts/tracts.parquet')
        """)

    # create spatial index on geom cols
    con.execute('CREATE INDEX buildings_spatial_idx ON temp_buildings USING RTREE (geometry)')
    con.execute('CREATE INDEX counties_spatial_idx ON temp_counties USING RTREE (geometry)')
    con.execute('CREATE INDEX tracts_spatial_idx ON temp_tracts USING RTREE (geometry)')

    # county level histograms
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
        TO 's3://carbonplan-ocr/intermediate/fire-risk/vector/{branch}/region_aggregation/county/county_summary_stats.parquet'
        (
                FORMAT 'parquet',
                COMPRESSION 'zstd',
                OVERWRITE_OR_IGNORE true);
        """)

    # tract level histograms - we should refactor this mostly shared SQL
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
        TO 's3://carbonplan-ocr/intermediate/fire-risk/vector/{branch}/region_aggregation/tract/tract_summary_stats.parquet'
        (
                FORMAT 'parquet',
                COMPRESSION 'zstd',
                OVERWRITE_OR_IGNORE true);
        """)


@click.command()
@click.option('-b', '--branch', help='data branch: [QA, prod]. Default QA')
def main(branch: str):
    create_summary_stats(branch=branch)


if __name__ == '__main__':
    main()
