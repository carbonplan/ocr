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
    buildings_path_glob: str,
):
    # Assume extensions & creds handled by caller.
    # tmp table for buildings
    con.execute(f"""
        CREATE TEMP TABLE buildings AS
        SELECT geometry,
        wind_risk_2011 as wind_risk_2011,
        wind_risk_2047 as wind_risk_2047
        FROM read_parquet('{buildings_path_glob}')
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

    # tmp table for block
    con.execute(f"""
        CREATE TEMP TABLE block AS
        SELECT GEOID, geometry
        FROM read_parquet('{block_path}')
        """)

    # create spatial index on geom cols
    con.execute('CREATE INDEX buildings_spatial_idx ON buildings USING RTREE (geometry)')
    con.execute('CREATE INDEX counties_spatial_idx ON county USING RTREE (geometry)')
    con.execute('CREATE INDEX tracts_spatial_idx ON tract USING RTREE (geometry)')
    con.execute('CREATE INDEX block_spatial_idx ON block USING RTREE (geometry)')


def custom_histogram_query(
    *,
    con: duckdb.DuckDBPyConnection,
    geo_table_name: str,
    summary_stats_path: UPath,
    hist_bins: list[int] | None = [0.01, 0.1, 1, 2, 3, 5, 7, 10, 15, 20, 100],
):
    # optional add if geo_table_name is county, we add a county Name to select.
    name_column = 'b.NAME as NAME,' if geo_table_name == 'county' else ''
    name_group_by = ', NAME' if geo_table_name == 'county' else ''

    hist_bin_padding = len(hist_bins)

    output_path = summary_stats_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    histogram_table = f""" COPY (
    SELECT
        b.GEOID as GEOID,
        {name_column}
        count(b.GEOID) as building_count,
        avg(a.wind_risk_2011) as mean_wind_risk_2011,
        avg(a.wind_risk_2047) as mean_wind_risk_2047,
        median(a.wind_risk_2011) as median_wind_risk_2011,
        median(a.wind_risk_2047) as median_wind_risk_2047,
        list_resize(map_values(histogram(a.wind_risk_2011, {hist_bins})), {hist_bin_padding}, 0) as wind_risk_2011,
        list_resize(map_values(histogram(a.wind_risk_2047, {hist_bins})), {hist_bin_padding}, 0) as wind_risk_2047,
        b.geometry as geometry
    FROM buildings a
    JOIN {geo_table_name} b ON ST_Intersects(a.geometry, b.geometry)
    GROUP BY GEOID, b.geometry{name_group_by})
        TO '{output_path}'
        (
                FORMAT 'parquet',
                COMPRESSION 'zstd',
                OVERWRITE_OR_IGNORE true);
    """
    con.execute(histogram_table)


def compute_regional_fire_wind_risk_statistics(config: OCRConfig):
    block_summary_stats_path = config.vector.block_summary_stats_uri
    tracts_summary_stats_path = config.vector.tracts_summary_stats_uri
    counties_summary_stats_path = config.vector.counties_summary_stats_uri
    buildings_path_glob = f'{config.vector.region_geoparquet_uri}/*.parquet'

    dataset = catalog.get_dataset('us-census-counties')
    counties_path = UPath(f's3://{dataset.bucket}/{dataset.prefix}')

    dataset = catalog.get_dataset('us-census-tracts')
    tracts_path = UPath(f's3://{dataset.bucket}/{dataset.prefix}')

    dataset = catalog.get_dataset('us-census-blocks')
    block_path = UPath(f's3://{dataset.bucket}/{dataset.prefix}')

    # The histogram syntax is kind of strange in duckdb, but since it's left-open, the first bin is values up to 0.01.
    hist_bins = [0.01, 0.1, 1, 2, 3, 5, 7, 10, 15, 20, 100]

    if config.debug:
        console.log(f'Using buildings path: {buildings_path_glob}')

    connection = duckdb.connect(database=':memory:')

    # Load required extensions (spatial + httpfs + aws) before any spatial ops or S3 reads
    install_load_extensions(aws=True, spatial=True, httpfs=True, con=connection)
    apply_s3_creds(con=connection)

    create_summary_stat_tmp_tables(
        con=connection,
        counties_path=counties_path,
        tracts_path=tracts_path,
        block_path=block_path,
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

    try:
        connection.close()
    except Exception:
        pass
