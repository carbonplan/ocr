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

    metadata_dict = config.vector.metadata_dict
    metadata_json = json.dumps(metadata_dict)

    con.execute(f"""
        CREATE TEMP TABLE {stats_table_name} AS
        SELECT
            GEOID,
            {extra_columns}
            building_count,
            rps_2011_mean,
            rps_2047_mean,
            bp_2011_mean,
            bp_2047_mean,
            crps_scott_mean,
            bp_2011_riley_mean,
            bp_2047_riley_mean,
            rps_2011_median,
            rps_2047_median,
            bp_2011_median,
            bp_2047_median,
            crps_scott_median,
            bp_2011_riley_median,
            bp_2047_riley_median,
            array_to_json(risk_score_2011_hist) as risk_score_2011_hist,
            array_to_json(risk_score_2047_hist) as risk_score_2047_hist,
            ST_X(ST_Centroid(geometry)) AS longitude,
            ST_Y(ST_Centroid(geometry)) AS latitude,
            geometry
        FROM read_parquet('{stats_parquet_path}')
    """)

    # Stream features row-by-row via fetchone() to avoid json_group_array overflow
    name_props = (
        "'STUSPS', STUSPS, 'NAME', NAME,"
        if stats_table_name == 'states'
        else "'NAME', NAME,"
        if stats_table_name == 'counties'
        else ''
    )

    feature_query = f"""
        SELECT json_object(
            'type', 'Feature',
            'properties', json_object(
                'GEOID', GEOID,
                {name_props}
                'building_count', building_count,
                'rps_2011_mean', rps_2011_mean,
                'rps_2047_mean', rps_2047_mean,
                'bp_2011_mean', bp_2011_mean,
                'bp_2047_mean', bp_2047_mean,
                'crps_scott_mean', crps_scott_mean,
                'bp_2011_riley_mean', bp_2011_riley_mean,
                'bp_2047_riley_mean', bp_2047_riley_mean,
                'rps_2011_median', rps_2011_median,
                'rps_2047_median', rps_2047_median,
                'bp_2011_median', bp_2011_median,
                'bp_2047_median', bp_2047_median,
                'crps_scott_median', crps_scott_median,
                'bp_2011_riley_median', bp_2011_riley_median,
                'bp_2047_riley_median', bp_2047_riley_median,
                'risk_score_2011_hist', risk_score_2011_hist,
                'risk_score_2047_hist', risk_score_2047_hist
            ),
            'geometry', json(ST_AsGeoJSON(geometry))
        )::VARCHAR as feature
        FROM {stats_table_name}
    """

    geojson_path = region_stats_path / 'stats.geojson'
    with geojson_path.open('w') as out:
        out.write('{"type":"FeatureCollection",')
        out.write(f'"metadata":{metadata_json},')
        out.write('"features":[')

        result = con.execute(feature_query)
        first = True
        while row := result.fetchone():
            if not first:
                out.write(',')
            out.write(row[0])
            first = False

        out.write(']}')

    # CSV output
    csv_path = region_stats_path / 'stats.csv'

    temp_csv_path = region_stats_path / 'stats_temp.csv'
    con.execute(
        f"""COPY (SELECT * EXCLUDE (geometry, longitude, latitude) FROM {stats_table_name}) TO '{temp_csv_path}';"""
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
