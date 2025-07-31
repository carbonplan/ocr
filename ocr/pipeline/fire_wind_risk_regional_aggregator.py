import duckdb
from upath import UPath

from ocr.console import console


def create_summary_stat_tmp_tables(
    *,
    con: duckdb.DuckDBPyConnection,
    counties_path: UPath,
    tracts_path: UPath,
    consolidated_buildings_path: UPath,
):
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

    # tmp table for geoms
    con.execute(f"""
        CREATE TEMP TABLE county AS
        SELECT NAME, geometry
        FROM read_parquet('{counties_path}')
        """)

    # tmp table for tracts
    con.execute(f"""
        CREATE TEMP TABLE tract AS
        SELECT GEOID as NAME, geometry
        FROM read_parquet('{tracts_path}')
        """)

    # create spatial index on geom cols
    con.execute('CREATE INDEX buildings_spatial_idx ON buildings USING RTREE (geometry)')
    con.execute('CREATE INDEX counties_spatial_idx ON county USING RTREE (geometry)')
    con.execute('CREATE INDEX tracts_spatial_idx ON tract USING RTREE (geometry)')
    return con


def custom_histogram_query(
    *,
    con: duckdb.DuckDBPyConnection,
    geo_table_name: str,
    aggregated_regions_prefix: str,
    hist_bins: list[int] | None = None,
):
    """The default duckdb histogram is left-open and right-closed, so to get counts of zero we need two create a counts of values that are exactly zero per county,
    then add them on to a histogram that excludes values of 0.
    """

    hist_bins = hist_bins or [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    console.log(f'Creating custom histogram query for {geo_table_name} with bins: {hist_bins}')

    # First temp table: zero counts by county
    zero_counts_query = f"""
    CREATE TEMP TABLE temp_zero_counts_{geo_table_name} AS
    SELECT
        b.NAME as NAME,
        count(CASE WHEN a.risk_2011_horizon_1 = 0 THEN 1 END) as zero_count_risk_2011_horizon_1,
        count(CASE WHEN a.risk_2011_horizon_15 = 0 THEN 1 END) as zero_count_risk_2011_horizon_15,
        count(CASE WHEN a.risk_2011_horizon_30 = 0 THEN 1 END) as zero_count_risk_2011_horizon_30,
        count(CASE WHEN a.risk_2047_horizon_1 = 0 THEN 1 END) as zero_count_risk_2047_horizon_1,
        count(CASE WHEN a.risk_2047_horizon_15 = 0 THEN 1 END) as zero_count_risk_2047_horizon_15,
        count(CASE WHEN a.risk_2047_horizon_30 = 0 THEN 1 END) as zero_count_risk_2047_horizon_30,
        count(CASE WHEN a.wind_risk_2011_horizon_1 = 0 THEN 1 END) as zero_count_wind_risk_2011_horizon_1,
        count(CASE WHEN a.wind_risk_2011_horizon_15 = 0 THEN 1 END) as zero_count_wind_risk_2011_horizon_15,
        count(CASE WHEN a.wind_risk_2011_horizon_30 = 0 THEN 1 END) as zero_count_wind_risk_2011_horizon_30,
        count(CASE WHEN a.wind_risk_2047_horizon_1 = 0 THEN 1 END) as zero_count_wind_risk_2047_horizon_1,
        count(CASE WHEN a.wind_risk_2047_horizon_15 = 0 THEN 1 END) as zero_count_wind_risk_2047_horizon_15,
        count(CASE WHEN a.wind_risk_2047_horizon_30 = 0 THEN 1 END) as zero_count_wind_risk_2047_horizon_30
    FROM buildings a
    JOIN {geo_table_name} b ON ST_Intersects(a.geometry, b.geometry)
    GROUP BY NAME
    """
    con.execute(zero_counts_query)

    # temp table #2 that excludes any 0 values and creates histograms.
    # filter out exact 0's and values greater then 100 (This shouldn't exist!)
    nonzero_hist_query = f"""
    CREATE TEMP TABLE temp_nonzero_histograms_{geo_table_name} AS
    SELECT
        b.NAME as NAME,
        count(a.risk_2011_horizon_1) as building_count,
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
    FROM buildings a
    JOIN {geo_table_name} b ON ST_Intersects(a.geometry, b.geometry)
    GROUP BY NAME, b.geometry

    """
    console.log(f'Executing nonzero histogram query for {geo_table_name}')

    con.execute(nonzero_hist_query)

    output_path = f'{aggregated_regions_prefix}/{geo_table_name}_summary_stats.parquet'

    # Now we merge the two temp tables together, the 0 counts table and the histograms that exclude 0.
    # duckdb has a func called `list_concat` for this.
    # We then write the result to parquet.
    merge_and_write = f""" COPY (
    SELECT
        h.NAME,
        h.building_count,
        h.avg_risk_2011_horizon_1,
        h.avg_risk_2011_horizon_15,
        h.avg_risk_2011_horizon_30,
        h.avg_risk_2047_horizon_1,
        h.avg_risk_2047_horizon_15,
        h.avg_risk_2047_horizon_30,
        h.avg_wind_risk_2011_horizon_1,
        h.avg_wind_risk_2011_horizon_15,
        h.avg_wind_risk_2011_horizon_30,
        h.avg_wind_risk_2047_horizon_1,
        h.avg_wind_risk_2047_horizon_15,
        h.avg_wind_risk_2047_horizon_30,
        list_concat([z.zero_count_risk_2011_horizon_1], h.nonzero_hist_risk_2011_horizon_1) as risk_2011_horizon_1,
        list_concat([z.zero_count_risk_2011_horizon_15], h.nonzero_hist_risk_2011_horizon_15) as risk_2011_horizon_15,
        list_concat([z.zero_count_risk_2011_horizon_30], h.nonzero_hist_risk_2011_horizon_30) as risk_2011_horizon_30,
        list_concat([z.zero_count_risk_2047_horizon_1], h.nonzero_hist_risk_2047_horizon_1) as risk_2047_horizon_1,
        list_concat([z.zero_count_risk_2047_horizon_15], h.nonzero_hist_risk_2047_horizon_15) as risk_2047_horizon_15,
        list_concat([z.zero_count_risk_2047_horizon_30], h.nonzero_hist_risk_2047_horizon_30) as risk_2047_horizon_30,
        list_concat([z.zero_count_wind_risk_2011_horizon_1], h.nonzero_hist_wind_risk_2011_horizon_1) as wind_risk_2011_horizon_1,
        list_concat([z.zero_count_wind_risk_2011_horizon_15], h.nonzero_hist_wind_risk_2011_horizon_15) as wind_risk_2011_horizon_15,
        list_concat([z.zero_count_wind_risk_2011_horizon_30], h.nonzero_hist_wind_risk_2011_horizon_30) as wind_risk_2011_horizon_30,
        list_concat([z.zero_count_wind_risk_2047_horizon_1], h.nonzero_hist_wind_risk_2047_horizon_1) as wind_risk_2047_horizon_1,
        list_concat([z.zero_count_wind_risk_2047_horizon_15], h.nonzero_hist_wind_risk_2047_horizon_15) as wind_risk_2047_horizon_15,
        list_concat([z.zero_count_wind_risk_2047_horizon_30], h.nonzero_hist_wind_risk_2047_horizon_30) as wind_risk_2047_horizon_30,
        h.geometry
    FROM temp_nonzero_histograms_{geo_table_name} h
    JOIN temp_zero_counts_{geo_table_name} z ON h.NAME = z.NAME)
        TO '{output_path}'
        (
                FORMAT 'parquet',
                COMPRESSION 'zstd',
                OVERWRITE_OR_IGNORE true);
    """
    con.execute(merge_and_write)
    console.log(f'Wrote summary statistics for {geo_table_name} to {output_path}')


def compute_regional_fire_wind_risk_statistics(
    counties_path: UPath,
    tracts_path: UPath,
    consolidated_buildings_path: UPath,
    aggregated_regions_prefix: str,
):
    con = duckdb.connect(database=':memory:')
    con.execute("""INSTALL SPATIAL; LOAD SPATIAL; INSTALL HTTPS; LOAD HTTPFS""")

    # The histogram syntax is kind of strange in duckdb, but since it's left-open, the first bin is values up to 10 (excluding zero from our earlier temp table filter).
    hist_bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    create_summary_stat_tmp_tables(
        con=con,
        counties_path=counties_path,
        tracts_path=tracts_path,
        consolidated_buildings_path=consolidated_buildings_path,
    )
    custom_histogram_query(
        con=con,
        geo_table_name='county',
        aggregated_regions_prefix=aggregated_regions_prefix,
        hist_bins=hist_bins,
    )
    custom_histogram_query(
        con=con,
        geo_table_name='tract',
        aggregated_regions_prefix=aggregated_regions_prefix,
        hist_bins=hist_bins,
    )
