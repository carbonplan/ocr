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
        rps_2011,
        rps_2047,
        bp_2011,
        bp_2047,
        crps_scott,
        bp_2011_riley,
        bp_2047_riley
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
    join_clause = f'JOIN {geo_table_name} b ON ST_Intersects(a.geometry, b.geometry)'

    if geo_table_name == 'county':
        name_column = 'b.NAME as NAME,'
        name_group_by = ', NAME'
    elif geo_table_name == 'state':
        name_column = 'b.STUSPS as STUSPS, b.NAME as NAME,'
        name_group_by = ', STUSPS, NAME'
    elif geo_table_name == 'nation':
        name_column = 'b.NAME as NAME,'
        name_group_by = ', NAME'
        # ignore spatial join for conus/nation
        join_clause = 'CROSS JOIN nation b'

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
        count(CASE WHEN a.rps_2011 = 0 THEN 1 END) as zero_count_rps_2011,
        count(CASE WHEN a.rps_2047 = 0 THEN 1 END) as zero_count_rps_2047,
        count(CASE WHEN a.bp_2011 = 0 THEN 1 END) as zero_count_bp_2011,
        count(CASE WHEN a.bp_2047 = 0 THEN 1 END) as zero_count_bp_2047,
        count(CASE WHEN a.crps_scott = 0 THEN 1 END) as zero_count_crps_scott,
        count(CASE WHEN a.bp_2011_riley = 0 THEN 1 END) as zero_count_bp_2011_riley,
        count(CASE WHEN a.bp_2047_riley = 0 THEN 1 END) as zero_count_bp_2047_riley
    FROM buildings a
    {join_clause}
    GROUP BY GEOID{name_group_by}
    """
    con.execute(zero_counts_query)

    nonzero_hist_query = f"""
    CREATE TEMP TABLE temp_nonzero_histograms_{geo_table_name} AS
    SELECT
        b.GEOID as GEOID,
        {name_column}
        count(b.GEOID) as building_count,

        avg(a.rps_2011) as rps_2011_mean,
        avg(a.rps_2047) as rps_2047_mean,
        avg(a.bp_2011) as bp_2011_mean,
        avg(a.bp_2047) as bp_2047_mean,
        avg(a.crps_scott) as crps_scott_mean,
        avg(a.bp_2011_riley) as bp_2011_riley_mean,
        avg(a.bp_2047_riley) as bp_2047_riley_mean,

        median(a.rps_2011) as rps_2011_median,
        median(a.rps_2047) as rps_2047_median,
        median(a.bp_2011) as bp_2011_median,
        median(a.bp_2047) as bp_2047_median,
        median(a.crps_scott) as crps_scott_median,
        median(a.bp_2011_riley) as bp_2011_riley_median,
        median(a.bp_2047_riley) as bp_2047_riley_median,

        list_resize(COALESCE(map_values(histogram(CASE WHEN a.rps_2011 <> 0 THEN a.rps_2011 END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_rps_2011,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.rps_2047 <> 0 THEN a.rps_2047 END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_rps_2047,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.bp_2011 <> 0 THEN a.bp_2011 END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_bp_2011,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.bp_2047 <> 0 THEN a.bp_2047 END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_bp_2047,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.crps_scott <> 0 THEN a.crps_scott END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_crps_scott,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.bp_2011_riley <> 0 THEN a.bp_2011_riley END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_bp_2011_riley,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.bp_2047_riley <> 0 THEN a.bp_2047_riley END, {hist_bins})), []), {hist_bin_padding}, 0) as nonzero_hist_bp_2047_riley,

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
        b.rps_2011_mean,
        b.rps_2047_mean,
        b.bp_2011_mean,
        b.bp_2047_mean,
        b.crps_scott_mean,
        b.bp_2011_riley_mean,
        b.bp_2047_riley_mean,
        b.rps_2011_median,
        b.rps_2047_median,
        b.bp_2011_median,
        b.bp_2047_median,
        b.crps_scott_median,
        b.bp_2011_riley_median,
        b.bp_2047_riley_median,
        list_concat([z.zero_count_rps_2011], b.nonzero_hist_rps_2011) as risk_score_2011_hist,
        list_concat([z.zero_count_rps_2047], b.nonzero_hist_rps_2047) as risk_score_2047_hist,
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
