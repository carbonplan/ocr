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
    consolidated_buildings_path: UPath,
):
    # Assume extensions & creds handled by caller.
    # tmp table for buildings
    con.execute(f"""
        CREATE TEMP TABLE buildings AS
        SELECT geometry,

        wind_risk_2011 as wind_risk_2011,
        wind_risk_2047 as wind_risk_2047,

        FROM read_parquet('{consolidated_buildings_path}')
        """)

    # tmp table for geoms
    con.execute(f"""
        CREATE TEMP TABLE county AS
        SELECT NAME, GEOID, geometry
        FROM read_parquet('{counties_path}')
        """)

    # tmp table for tracts
    con.execute(f"""
        CREATE TEMP TABLE tract AS
        SELECT GEOID, geometry
        FROM read_parquet('{tracts_path}')
        """)

    # create spatial index on geom cols
    con.execute('CREATE INDEX buildings_spatial_idx ON buildings USING RTREE (geometry)')
    con.execute('CREATE INDEX counties_spatial_idx ON county USING RTREE (geometry)')
    con.execute('CREATE INDEX tracts_spatial_idx ON tract USING RTREE (geometry)')


def custom_histogram_query(
    *,
    con: duckdb.DuckDBPyConnection,
    geo_table_name: str,
    summary_stats_path: UPath,
    hist_bins: list[int] | None = [0.01, 0.1, 1, 2, 3, 5, 7, 10, 15, 20, 100],
):
    """The default duckdb histogram is left-open and right-closed, so to get counts of zero we need two create a counts of values that are exactly zero per county,
    then add them on to a histogram that excludes values of 0.
    """
    # optional add if geo_table_name is county, we add a county Name to select.
    name_column = 'b.NAME as NAME,' if geo_table_name == 'county' else ''
    name_group_by = ', NAME' if geo_table_name == 'county' else ''

    # Connection, extensions, and credentials are managed by caller.

    # First temp table: zero counts by county.
    hist_bin_padding = len(hist_bins)
    zero_counts_query = f"""
    CREATE TEMP TABLE temp_zero_counts_{geo_table_name} AS
    SELECT
        b.GEOID as GEOID,
        {name_column}
        count(CASE WHEN a.wind_risk_2011 = 0 THEN 1 END) as zero_count_wind_risk_2011,
        count(CASE WHEN a.wind_risk_2047 = 0 THEN 1 END) as zero_count_wind_risk_2047,

    FROM buildings a
    JOIN {geo_table_name} b ON ST_Intersects(a.geometry, b.geometry)
    GROUP BY GEOID{name_group_by}
    """
    con.execute(zero_counts_query)

    # temp table #2 that excludes any 0 values and creates histograms.
    # filter out exact 0's and values greater then 100 (This shouldn't exist!)
    nonzero_hist_query = f"""
    CREATE TEMP TABLE temp_nonzero_histograms_{geo_table_name} AS
    SELECT
        b.GEOID as GEOID,
        {name_column}
        count(b.GEOID) as building_count,
        avg(a.wind_risk_2011) as mean_wind_risk_2011,
        avg(a.wind_risk_2047) as mean_wind_risk_2047,
        median(a.wind_risk_2011) as median_wind_risk_2011,
        median(a.wind_risk_2047) as median_wind_risk_2047,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.wind_risk_2011 <> 0 THEN a.wind_risk_2011 END, {hist_bins})),[]), {hist_bin_padding}, 0) as nonzero_hist_wind_risk_2011,
        list_resize(COALESCE(map_values(histogram(CASE WHEN a.wind_risk_2047 <> 0 THEN a.wind_risk_2047 END, {hist_bins})),[]), {hist_bin_padding}, 0) as nonzero_hist_wind_risk_2047,

        b.geometry as geometry
    FROM buildings a
    JOIN {geo_table_name} b ON ST_Intersects(a.geometry, b.geometry)
    GROUP BY GEOID, b.geometry{name_group_by}

    """

    con.execute(nonzero_hist_query)

    output_path = summary_stats_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Now we merge the two temp tables together, the 0 counts table and the histograms that exclude 0.
    # duckdb has a func called `list_concat` for this.
    # We then write the result to parquet.
    merge_and_write = f""" COPY (
    SELECT
        b.GEOID,
        {name_column}
        b.building_count,
        b.mean_wind_risk_2011,
        b.mean_wind_risk_2047,
        b.median_wind_risk_2011,
        b.median_wind_risk_2047,
        list_concat([z.zero_count_wind_risk_2011], b.nonzero_hist_wind_risk_2011) as wind_risk_2011,
        list_concat([z.zero_count_wind_risk_2047], b.nonzero_hist_wind_risk_2047) as wind_risk_2047,

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
    tracts_summary_stats_path = config.vector.tracts_summary_stats_uri
    counties_summary_stats_path = config.vector.counties_summary_stats_uri
    consolidated_buildings_path = config.vector.building_geoparquet_uri

    dataset = catalog.get_dataset('us-census-counties')
    counties_path = UPath(f's3://{dataset.bucket}/{dataset.prefix}')

    dataset = catalog.get_dataset('us-census-tracts')
    tracts_path = UPath(f's3://{dataset.bucket}/{dataset.prefix}')

    # The histogram syntax is kind of strange in duckdb, but since it's left-open, the first bin is values up to 0.01 (excluding zero from our earlier temp table filter).
    hist_bins = [0.01, 0.1, 1, 2, 3, 5, 7, 10, 15, 20, 100]

    if config.debug:
        console.log(f'Using consolidated buildings path: {consolidated_buildings_path}')

    connection = duckdb.connect(database=':memory:')

    # Load required extensions (spatial + httpfs + aws) before any spatial ops or S3 reads
    install_load_extensions(aws=True, spatial=True, httpfs=True, con=connection)
    apply_s3_creds(con=connection)

    create_summary_stat_tmp_tables(
        con=connection,
        counties_path=counties_path,
        tracts_path=tracts_path,
        consolidated_buildings_path=consolidated_buildings_path,
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

    try:
        connection.close()
    except Exception:
        pass
