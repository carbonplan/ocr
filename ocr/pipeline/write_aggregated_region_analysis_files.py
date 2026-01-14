import json

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

    metadata_dict = config.metadata_dict
    metadata_json = json.dumps(metadata_dict)

    con.execute(f"""
        CREATE TEMP TABLE {stats_table_name} AS
        SELECT
            GEOID,
            {extra_columns}
            building_count,
            mean_wind_risk_2011 as avg_wind_risk_2011,
            mean_wind_risk_2047 as avg_wind_risk_2047,
            mean_burn_probability_2011 as avg_burn_probability_2011,
            mean_burn_probability_2047 as avg_burn_probability_2047,
            mean_conditional_risk_usfs as avg_conditional_risk_usfs,
            mean_burn_probability_usfs_2011 as avg_burn_probability_usfs_2011,
            mean_burn_probability_usfs_2047 as avg_burn_probability_usfs_2047,
            median_wind_risk_2011,
            median_wind_risk_2047,
            median_burn_probability_2011,
            median_burn_probability_2047,
            median_conditional_risk_usfs,
            median_burn_probability_usfs_2011,
            median_burn_probability_usfs_2047,
            array_to_json(wind_risk_2011_hist) as wind_risk_2011_hist,
            array_to_json(wind_risk_2047_hist) as wind_risk_2047_hist,
            array_to_json(burn_probability_2011_hist) as burn_probability_2011_hist,
            array_to_json(burn_probability_2047_hist) as burn_probability_2047_hist,
            array_to_json(conditional_risk_usfs_hist) as conditional_risk_usfs_hist,
            array_to_json(burn_probability_usfs_2011_hist) as burn_probability_usfs_2011_hist,
            array_to_json(burn_probability_usfs_2047_hist) as burn_probability_usfs_2047_hist,
            ST_X(ST_Centroid(geometry)) AS centroid_longitude,
            ST_Y(ST_Centroid(geometry)) AS centroid_latitude,
            geometry
        FROM read_parquet('{stats_parquet_path}')
    """)

    # create geojson with metadata at top level
    con.execute(f"""
        COPY (
            SELECT json_object(
                'type', 'FeatureCollection',
                'metadata', '{metadata_json}'::JSON,
                'features', (
                    SELECT json_group_array(
                        json_object(
                            'type', 'Feature',
                            'properties', json_object(
                                'GEOID', GEOID,
                                {"'STUSPS', STUSPS, 'NAME', NAME," if stats_table_name == 'states' else "'NAME', NAME," if stats_table_name == 'counties' else ''}
                                'building_count', building_count,
                                'avg_wind_risk_2011', avg_wind_risk_2011,
                                'avg_wind_risk_2047', avg_wind_risk_2047,
                                'avg_burn_probability_2011', avg_burn_probability_2011,
                                'avg_burn_probability_2047', avg_burn_probability_2047,
                                'avg_conditional_risk_usfs', avg_conditional_risk_usfs,
                                'avg_burn_probability_usfs_2011', avg_burn_probability_usfs_2011,
                                'avg_burn_probability_usfs_2047', avg_burn_probability_usfs_2047,
                                'median_wind_risk_2011', median_wind_risk_2011,
                                'median_wind_risk_2047', median_wind_risk_2047,
                                'median_burn_probability_2011', median_burn_probability_2011,
                                'median_burn_probability_2047', median_burn_probability_2047,
                                'median_conditional_risk_usfs', median_conditional_risk_usfs,
                                'median_burn_probability_usfs_2011', median_burn_probability_usfs_2011,
                                'median_burn_probability_usfs_2047', median_burn_probability_usfs_2047,
                                'wind_risk_2011_hist', wind_risk_2011_hist,
                                'wind_risk_2047_hist', wind_risk_2047_hist,
                                'burn_probability_2011_hist', burn_probability_2011_hist,
                                'burn_probability_2047_hist', burn_probability_2047_hist,
                                'conditional_risk_usfs_hist', conditional_risk_usfs_hist,
                                'burn_probability_usfs_2011_hist', burn_probability_usfs_2011_hist,
                                'burn_probability_usfs_2047_hist', burn_probability_usfs_2047_hist
                            ),
                            'geometry', json(ST_AsGeoJSON(geometry))
                        )
                    )
                    FROM {stats_table_name}
                )
            )::VARCHAR as content
        ) TO '{region_stats_path}/stats.geojson' (FORMAT csv, HEADER false, QUOTE '');
    """)

    csv_path = region_stats_path / 'stats.csv'

    temp_csv_path = region_stats_path / 'stats_temp.csv'
    con.execute(
        f"""COPY (SELECT * EXCLUDE (geometry, centroid_longitude, centroid_latitude) FROM {stats_table_name}) TO '{temp_csv_path}';"""
    )

    csv_content = temp_csv_path.read_text()

    metadata_header = '\n'.join([f'# {key}: {value}' for key, value in metadata_dict.items()])

    csv_path.write_text(f'{metadata_header}\n{csv_content}')

    temp_csv_path.unlink()


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
