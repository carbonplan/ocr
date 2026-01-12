import duckdb
from upath import UPath

from ocr import catalog
from ocr.config import OCRConfig
from ocr.console import console
from ocr.utils import apply_s3_creds, install_load_extensions


def create_summary_stat_tmp_tables(
    *,
    con: duckdb.DuckDBPyConnection,
    counties_path: UPath,
    tracts_path: UPath,
    block_path: UPath,
    states_path: UPath,
    nation_path: UPath,
    buildings_path_glob: str,
):
    con.execute(f"""
        CREATE TEMP TABLE buildings AS
        SELECT geometry,
        wind_risk_2011,
        wind_risk_2047,
        burn_probability_2011,
        burn_probability_2047,
        conditional_risk_usfs,
        burn_probability_usfs_2011,
        burn_probability_usfs_2047
        FROM read_parquet('{buildings_path_glob}')
        """)

    con.execute(f"""
        CREATE TEMP TABLE county AS
        SELECT NAME, GEOID, geometry
        FROM read_parquet('{counties_path}')
        """)

    con.execute(f"""
        CREATE TEMP TABLE tract AS
        SELECT GEOID, geometry
        FROM read_parquet('{tracts_path}')
        """)

    con.execute(f"""
        CREATE TEMP TABLE block AS
        SELECT GEOID, geometry
        FROM read_parquet('{block_path}')
        """)

    con.execute(f"""
        CREATE TEMP TABLE state AS
        SELECT GEOID, STUSPS, NAME, geometry
        FROM read_parquet('{states_path}')
        """)

    con.execute(f"""
        CREATE TEMP TABLE nation AS
        SELECT GEOID, NAME, geometry
        FROM read_parquet('{nation_path}')
        """)

    con.execute('CREATE INDEX buildings_spatial_idx ON buildings USING RTREE (geometry)')
    con.execute('CREATE INDEX counties_spatial_idx ON county USING RTREE (geometry)')
    con.execute('CREATE INDEX tracts_spatial_idx ON tract USING RTREE (geometry)')
    con.execute('CREATE INDEX block_spatial_idx ON block USING RTREE (geometry)')
    con.execute('CREATE INDEX states_spatial_idx ON state USING RTREE (geometry)')
    con.execute('CREATE INDEX nation_spatial_idx ON nation USING RTREE (geometry)')


def custom_histogram_query(
    *,
    con: duckdb.DuckDBPyConnection,
    geo_table_name: str,
    summary_stats_path: UPath,
    hist_bins: list[int] | None = [0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.5, 1, 3, 100],
):
    if geo_table_name == 'county':
        name_column = 'b.NAME as NAME,'
        name_group_by = ', NAME'
    elif geo_table_name == 'state':
        name_column = 'b.STUSPS as STUSPS, b.NAME as NAME,'
        name_group_by = ', STUSPS, NAME'
    elif geo_table_name == 'nation':
        name_column = 'b.NAME as NAME,'
        name_group_by = ', NAME'
    else:
        name_column = ''
        name_group_by = ''

    hist_bin_padding = len(hist_bins)

    output_path = summary_stats_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zero_counts_query = f"""
    CREATE TEMP TABLE temp_zero_counts_{geo_table_name} AS
    SELECT
        b.GEOID as GEOID,
        {name_column}
        count(CASE WHEN a.wind_risk_2011 = 0 THEN 1 END) as zero_count_wind_risk_2011,
        count(CASE WHEN a.wind_risk_2047 = 0 THEN 1 END) as zero_count_wind_risk_2047,
        count(CASE WHEN a.burn_probability_2011 = 0 THEN 1 END) as zero_count_burn_probability_2011,
        count(CASE WHEN a.burn_probability_2047 = 0 THEN 1 END) as zero_count_burn_probability_2047,
        count(CASE WHEN a.conditional_risk_usfs = 0 THEN 1 END) as zero_count_conditional_risk_usfs,
        count(CASE WHEN a.burn_probability_usfs_2011 = 0 THEN 1 END) as zero_count_burn_probability_usfs_2011,
        count(CASE WHEN a.burn_probability_usfs_2047 = 0 THEN 1 END) as zero_count_burn_probability_usfs_2047
    FROM buildings a
    JOIN {geo_table_name} b ON ST_Intersects(a.geometry, b.geometry)
    GROUP BY GEOID{name_group_by}
    """
    con.execute(zero_counts_query)

    nonzero_hist_query = f"""
    CREATE TEMP TABLE temp_nonzero_histograms_{geo_table_name} AS
    SELECT
        b.GEOID as GEOID,
        {name_column}
        count(b.GEOID) as building_count,

        avg(a.wind_risk_2011) as mean_wind_risk_2011,
        avg(a.wind_risk_2047) as mean_wind_risk_2047,
        avg(a.burn_probability_2011) as mean_burn_probability_2011,
        avg(a.burn_probability_2047) as mean_burn_probability_2047,
        avg(a.conditional_risk_usfs) as mean_conditional_risk_usfs,
        avg(a.burn_probability_usfs_2011) as mean_burn_probability_usfs_2011,
        avg(a.burn_probability_usfs_2047) as mean_burn_probability_usfs_2047,

        median(a.wind_risk_2011) as median_wind_risk_2011,
        median(a.wind_risk_2047) as median_wind_risk_2047,
        median(a.burn_probability_2011) as median_burn_probability_2011,
        median(a.burn_probability_2047) as median_burn_probability_2047,
        median(a.conditional_risk_usfs) as median_conditional_risk_usfs,
        median(a.burn_probability_usfs_2011) as median_burn_probability_usfs_2011,
        median(a.burn_probability_usfs_2047) as median_burn_probability_usfs_2047,

        list_resize(COALESCE(map_values(histogram(CASE WHEN a.wind_risk_2011 <> 0 THEN a.wind_risk_2011 END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_wind_risk_2011,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.wind_risk_2047 <> 0 THEN a.wind_risk_2047 END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_wind_risk_2047,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.burn_probability_2011 <> 0 THEN a.burn_probability_2011 END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_burn_probability_2011,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.burn_probability_2047 <> 0 THEN a.burn_probability_2047 END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_burn_probability_2047,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.conditional_risk_usfs <> 0 THEN a.conditional_risk_usfs END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_conditional_risk_usfs,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.burn_probability_usfs_2011 <> 0 THEN a.burn_probability_usfs_2011 END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_burn_probability_usfs_2011,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.burn_probability_usfs_2047 <> 0 THEN a.burn_probability_usfs_2047 END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_burn_probability_usfs_2047,

        b.geometry as geometry
    FROM buildings a
    JOIN {geo_table_name} b ON ST_Intersects(a.geometry, b.geometry)
    GROUP BY GEOID, b.geometry{name_group_by}
    """
    con.execute(nonzero_hist_query)

    merge_and_write = f""" COPY (
    SELECT
        b.GEOID,
        {name_column}
        b.building_count,
        b.mean_wind_risk_2011,
        b.mean_wind_risk_2047,
        b.mean_burn_probability_2011,
        b.mean_burn_probability_2047,
        b.mean_conditional_risk_usfs,
        b.mean_burn_probability_usfs_2011,
        b.mean_burn_probability_usfs_2047,
        b.median_wind_risk_2011,
        b.median_wind_risk_2047,
        b.median_burn_probability_2011,
        b.median_burn_probability_2047,
        b.median_conditional_risk_usfs,
        b.median_burn_probability_usfs_2011,
        b.median_burn_probability_usfs_2047,
        list_concat([z.zero_count_wind_risk_2011], b.nonzero_hist_wind_risk_2011) as wind_risk_2011_hist,
        list_concat([z.zero_count_wind_risk_2047], b.nonzero_hist_wind_risk_2047) as wind_risk_2047_hist,
        list_concat([z.zero_count_burn_probability_2011], b.nonzero_hist_burn_probability_2011) as burn_probability_2011_hist,
        list_concat([z.zero_count_burn_probability_2047], b.nonzero_hist_burn_probability_2047) as burn_probability_2047_hist,
        list_concat([z.zero_count_conditional_risk_usfs], b.nonzero_hist_conditional_risk_usfs) as conditional_risk_usfs_hist,
        list_concat([z.zero_count_burn_probability_usfs_2011], b.nonzero_hist_burn_probability_usfs_2011) as burn_probability_usfs_2011_hist,
        list_concat([z.zero_count_burn_probability_usfs_2047], b.nonzero_hist_burn_probability_usfs_2047) as burn_probability_usfs_2047_hist,
        b.geometry
    FROM temp_nonzero_histograms_{geo_table_name} b
    JOIN temp_zero_counts_{geo_table_name} z ON b.GEOID = z.GEOID)
        TO '{output_path}'
        (
                FORMAT 'parquet',
                COMPRESSION 'zstd',
                OVERWRITE_OR_IGNORE true);
    """
    con.execute(merge_and_write)


def compute_regional_fire_wind_risk_statistics(config: OCRConfig):
    block_summary_stats_path = config.vector.block_summary_stats_uri
    tracts_summary_stats_path = config.vector.tracts_summary_stats_uri
    counties_summary_stats_path = config.vector.counties_summary_stats_uri
    states_summary_stats_path = config.vector.states_summary_stats_uri
    nation_summary_stats_path = config.vector.nation_summary_stats_uri
    buildings_path_glob = f'{config.vector.region_geoparquet_uri}/*.parquet'

    dataset = catalog.get_dataset('us-census-counties')
    counties_path = UPath(f's3://{dataset.bucket}/{dataset.prefix}')

    dataset = catalog.get_dataset('us-census-tracts')
    tracts_path = UPath(f's3://{dataset.bucket}/{dataset.prefix}')

    dataset = catalog.get_dataset('us-census-blocks')
    block_path = UPath(f's3://{dataset.bucket}/{dataset.prefix}')

    dataset = catalog.get_dataset('us-census-states')
    states_path = UPath(f's3://{dataset.bucket}/{dataset.prefix}')

    dataset = catalog.get_dataset('us-census-nation')
    nation_path = UPath(f's3://{dataset.bucket}/{dataset.prefix}')

    hist_bins = [0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.5, 1, 3, 100]
    if config.debug:
        console.log(f'Using buildings path: {buildings_path_glob}')

    connection = duckdb.connect(database=':memory:')

    install_load_extensions(aws=True, spatial=True, httpfs=True, con=connection)
    apply_s3_creds(con=connection)

    create_summary_stat_tmp_tables(
        con=connection,
        counties_path=counties_path,
        tracts_path=tracts_path,
        block_path=block_path,
        states_path=states_path,
        nation_path=nation_path,
        buildings_path_glob=buildings_path_glob,
    )

    if config.debug:
        console.log('Computing county summary statistics')
    custom_histogram_query(
        con=connection,
        geo_table_name='county',
        summary_stats_path=counties_summary_stats_path,
        hist_bins=hist_bins,
    )
    if config.debug:
        console.log(f'Wrote summary statistics for county to {counties_summary_stats_path}')

    if config.debug:
        console.log('Computing tract summary statistics')
    custom_histogram_query(
        con=connection,
        geo_table_name='tract',
        summary_stats_path=tracts_summary_stats_path,
        hist_bins=hist_bins,
    )
    if config.debug:
        console.log(f'Wrote summary statistics for tract to {tracts_summary_stats_path}')

    if config.debug:
        console.log('Computing block summary statistics')
    custom_histogram_query(
        con=connection,
        geo_table_name='block',
        summary_stats_path=block_summary_stats_path,
        hist_bins=hist_bins,
    )
    if config.debug:
        console.log(f'Wrote summary statistics for block to {block_summary_stats_path}')

    if config.debug:
        console.log('Computing state summary statistics')
    custom_histogram_query(
        con=connection,
        geo_table_name='state',
        summary_stats_path=states_summary_stats_path,
        hist_bins=hist_bins,
    )
    if config.debug:
        console.log(f'Wrote summary statistics for state to {states_summary_stats_path}')

    if config.debug:
        console.log('Computing nation summary statistics')
    custom_histogram_query(
        con=connection,
        geo_table_name='nation',
        summary_stats_path=nation_summary_stats_path,
        hist_bins=hist_bins,
    )
    if config.debug:
        console.log(f'Wrote summary statistics for nation to {nation_summary_stats_path}')

    connection.close()
