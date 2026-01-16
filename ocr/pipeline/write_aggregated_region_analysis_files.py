import duckdb
from upath import UPath

from ocr.config import OCRConfig
from ocr.console import console
from ocr.utils import apply_s3_creds, install_load_extensions


def write_stats_table(
    *,
    con: duckdb.DuckDBPyConnection,
    config: OCRConfig,
    stats_parquet_path: UPath,
    stats_table_name: str,
):
    region_analysis_path = config.vector.aggregated_region_analysis_uri
    region_stats_path = region_analysis_path / stats_table_name
    region_stats_path.mkdir(parents=True, exist_ok=True)

    extra_columns = ''
    if stats_table_name == 'states':
        extra_columns = 'STUSPS, NAME,'  # the state summary stats also has state name and abbv.
    elif stats_table_name == 'counties':
        extra_columns = 'NAME,'

    con.execute(f"""
        CREATE TEMP TABLE {stats_table_name} AS
        SELECT
            GEOID,
            {extra_columns}
            building_count,
            mean_rps_2011 as rps_2011_mean,
            mean_rps_2047 as rps_2047_mean,
            mean_bp_2011 as bp_2011_mean,
            mean_bp_2047 as bp_2047_mean,
            mean_crps_scott as crps_scott_mean,
            mean_bp_2011_riley as bp_2011_riley_mean,
            mean_bp_2047_riley as bp_2011_riley_mean,
            rps_2011_median,
            rps_2047_median,
            bp_2011_median,
            bp_2047_median,
            crps_scott_median,
            bp_2011_riley_median,
            bp_2047_riley_median,
            array_to_json(risk_score_2011_hist) as risk_score_2011_hist,
            array_to_json(risk_score_2047_hist) as risk_score_2047_hist,
            array_to_json(bp_2011_hist) as bp_2011_hist,
            array_to_json(bp_2047_hist) as bp_2047_hist,
            array_to_json(crps_scott_hist) as crps_scott_hist,
            array_to_json(bp_2011_riley_hist) as bp_2011_riley_hist,
            array_to_json(bp_2047_riley_hist) as bp_2047_riley_hist,
            ST_X(ST_Centroid(geometry)) AS longitude,
            ST_Y(ST_Centroid(geometry)) AS latitude,
            geometry
        FROM read_parquet('{stats_parquet_path}')
    """)

    con.execute(
        f"""COPY (SELECT * EXCLUDE (longitude, latitude) FROM {stats_table_name}) TO '{region_stats_path}/stats.geojson' WITH (FORMAT GDAL, DRIVER 'GeoJSON', LAYER_NAME 'STATS', OVERWRITE_OR_IGNORE true);"""
    )

    con.execute(
        f"""COPY (SELECT * EXCLUDE geometry FROM {stats_table_name}) TO '{region_stats_path}/stats.csv';"""
    )


def write_aggregated_region_analysis_files(config: OCRConfig):
    block_summary_stats_path = config.vector.block_summary_stats_uri
    tracts_summary_stats_path = config.vector.tracts_summary_stats_uri
    counties_summary_stats_path = config.vector.counties_summary_stats_uri
    states_summary_stats_path = config.vector.states_summary_stats_uri
    nation_summary_stats_path = config.vector.nation_summary_stats_uri

    connection = duckdb.connect(database=':memory:')

    install_load_extensions(aws=True, spatial=True, httpfs=True, con=connection)
    apply_s3_creds(con=connection)

    if config.debug:
        console.log('Writing aggregated region analysis files for census blocks.')
    write_stats_table(
        con=connection,
        config=config,
        stats_parquet_path=block_summary_stats_path,
        stats_table_name='block',
    )

    if config.debug:
        console.log('Writing aggregated region analysis files for counties.')
    write_stats_table(
        con=connection,
        config=config,
        stats_parquet_path=counties_summary_stats_path,
        stats_table_name='counties',
    )

    if config.debug:
        console.log('Writing aggregated region analysis files for census tracts.')
    write_stats_table(
        con=connection,
        config=config,
        stats_parquet_path=tracts_summary_stats_path,
        stats_table_name='tracts',
    )

    if config.debug:
        console.log('Writing aggregated region analysis files for states.')
    write_stats_table(
        con=connection,
        config=config,
        stats_parquet_path=states_summary_stats_path,
        stats_table_name='states',
    )

    if config.debug:
        console.log('Writing aggregated region analysis file for CONUS.')
    write_stats_table(
        con=connection,
        config=config,
        stats_parquet_path=nation_summary_stats_path,
        stats_table_name='nation',
    )

    try:
        connection.close()
    except Exception:
        pass
