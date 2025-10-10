"""Creates summary stats for spatial aggregated regions (counties, census tracts) and writes those to common file formats"""

import duckdb
from upath import UPath

from ocr import catalog
from ocr.config import OCRConfig
from ocr.console import console
from ocr.utils import apply_s3_creds, install_load_extensions


def write_stats_table(
    *,
    con: duckdb.DuckDBPyConnection,
    config: OCRConfig,
    region_path: UPath,
    stats_table_name: str,
    hist_bins: list | None = [5, 10, 15, 20, 25, 100],
):
    region_analysis_path = config.vector.aggregated_region_analysis_uri
    consolidated_buildings_path = config.vector.building_geoparquet_uri

    region_stats_path = region_analysis_path / stats_table_name

    region_stats_path.mkdir(parents=True, exist_ok=True)

    con.execute(f"""
        CREATE TEMP TABLE {stats_table_name} AS
        SELECT a.GEOID,
            COUNT(b.wind_risk_2011) as building_count,

            AVG(b.burn_probability_2011) as avg_burn_probability_2011,
            AVG(b.burn_probability_2047) as avg_burn_probability_2047,
            AVG(b.wind_risk_2011) as avg_wind_risk_2011,
            AVG(b.wind_risk_2047) as avg_wind_risk_2047,

            MEDIAN(b.burn_probability_2011) as median_burn_probability_2011,
            MEDIAN(b.burn_probability_2047) as median_burn_probability_2047,
            MEDIAN(b.wind_risk_2011) as median_wind_risk_2011,
            MEDIAN(b.wind_risk_2047) as median_wind_risk_2047,


            quantile_cont(b.burn_probability_2011, 0.90) as p90_burn_probability_2011,
            quantile_cont(b.burn_probability_2011, 0.95) as p95_burn_probability_2011,
            quantile_cont(b.burn_probability_2011, 0.99) as p99_burn_probability_2011,

            quantile_cont(b.burn_probability_2047, 0.90) as p90_burn_probability_2047,
            quantile_cont(b.burn_probability_2047, 0.95) as p95_burn_probability_2047,
            quantile_cont(b.burn_probability_2047, 0.99) as p99_burn_probability_2047,

            quantile_cont(b.wind_risk_2011, 0.90) as p90_wind_risk_2011,
            quantile_cont(b.wind_risk_2011, 0.95) as p95_wind_risk_2011,
            quantile_cont(b.wind_risk_2011, 0.99) as p99_wind_risk_2011,

            quantile_cont(b.wind_risk_2047, 0.90) as p90_wind_risk_2047,
            quantile_cont(b.wind_risk_2047, 0.95) as p95_wind_risk_2047,
            quantile_cont(b.wind_risk_2047, 0.99) as p99_wind_risk_2047,

            -- we have to cast the histogram from HUGEINT[] to array_json since gdal/json does not support HUGEINT[]

            array_to_json(list_concat(
                [count(CASE WHEN b.burn_probability_2011 = 0 THEN 1 END)],
                map_values(histogram(CASE WHEN b.burn_probability_2011 <> 0 THEN b.burn_probability_2011 END, {hist_bins}))
            )) as burn_probability_2011_hist,

            array_to_json(list_concat(
                [count(CASE WHEN b.burn_probability_2047 = 0 THEN 1 END)],
                map_values(histogram(CASE WHEN b.burn_probability_2047 <> 0 THEN b.burn_probability_2047 END, {hist_bins}))
            )) as burn_probability_2047_hist,

            array_to_json(list_concat(
                [count(CASE WHEN b.wind_risk_2011 = 0 THEN 1 END)],
                map_values(histogram(CASE WHEN b.wind_risk_2011 <> 0 THEN b.wind_risk_2011 END, {hist_bins}))
            )) as wind_risk_2011_hist,

            array_to_json(list_concat(
                [count(CASE WHEN b.wind_risk_2047 = 0 THEN 1 END)],
                map_values(histogram(CASE WHEN b.wind_risk_2047 <> 0 THEN b.wind_risk_2047 END, {hist_bins}))
            )) as wind_risk_2047_hist,

            a.geometry
        FROM read_parquet('{region_path}') a
        JOIN read_parquet('{consolidated_buildings_path}') b
            ON ST_Intersects(a.geometry, b.geometry)
        GROUP BY a.GEOID, a.geometry ;
    """)

    # Write Geoparquet
    con.execute(f"""COPY {stats_table_name} TO '{region_stats_path}/stats.parquet' (
        FORMAT 'parquet',
        COMPRESSION 'zstd',
        OVERWRITE_OR_IGNORE true);""")

    # Write GeoJSON
    con.execute(
        f"""COPY {stats_table_name} TO '{region_stats_path}/stats.geojson' WITH (FORMAT GDAL, DRIVER 'GeoJSON', LAYER_NAME 'STATS', OVERWRITE_OR_IGNORE true);"""
    )

    # Write CSV
    con.execute(
        f"""COPY (SELECT * EXCLUDE geometry FROM {stats_table_name}) TO '{region_stats_path}/stats.csv';"""
    )

    # this might break, shapefiles are terrible
    # IOException: IO Error: GDAL Error (1): Failed to create file
    # con.execute(f"""COPY {stats_table_name} TO '{region_stats_path}/stats.shp' (FORMAT GDAL, DRIVER 'ESRI Shapefile');""")


def write_aggregated_region_analysis_files(config: OCRConfig):
    hist_bins = [0.01, 0.1, 1, 2, 3, 5, 7, 10, 15, 20, 100]

    counties_dataset = catalog.get_dataset('us-census-counties')
    counties_path = UPath(f's3://{counties_dataset.bucket}/{counties_dataset.prefix}')

    tracts_dataset = catalog.get_dataset('us-census-tracts')
    tracts_path = UPath(f's3://{tracts_dataset.bucket}/{tracts_dataset.prefix}')

    connection = duckdb.connect(database=':memory:')

    # Load required extensions (spatial + httpfs + aws) before any spatial ops or S3 reads
    install_load_extensions(aws=True, spatial=True, httpfs=True, con=connection)
    apply_s3_creds(con=connection)

    if config.debug:
        console.log('Writing aggregated region analysis files for counties.')
    write_stats_table(
        con=connection,
        config=config,
        region_path=counties_path,
        stats_table_name='counties',
        hist_bins=hist_bins,
    )
    if config.debug:
        console.log('Writing aggregated region analysis files for tracts.')
    write_stats_table(
        con=connection,
        config=config,
        region_path=tracts_path,
        stats_table_name='tracts',
        hist_bins=hist_bins,
    )
