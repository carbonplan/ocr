"""For each (counties, census tract), write analysis files in multiple file formats"""

import gzip
from typing import Literal

import boto3
import duckdb
from rich.progress import track
from upath import UPath

from ocr import catalog
from ocr.config import OCRConfig
from ocr.console import console
from ocr.types import RegionType
from ocr.utils import apply_s3_creds, install_load_extensions


def _modify_headers(bucket: str, prefix: str, content_type: Literal['text/csv', 'text/json']):
    """Updates the file headers in S3 so that compressed data will be automatically uncompressed by the browser on download.
    If the data is geojson, we are also modify it by compressing it because the duckdb/GDAL driver can't write compressed geojson."""

    s3_client = boto3.client('s3')

    if content_type == 'text/json':
        # download the geojson contents, compress and re-upload with boto3
        response = s3_client.get_object(Bucket=bucket, Key=prefix)
        src_data = response['Body'].read()
        compressed_data = gzip.compress(src_data)

        # upload compressed version back to s3
        s3_client.put_object(
            Bucket=bucket,
            Key=prefix,
            Body=compressed_data,
            ContentType=content_type,
            ContentEncoding='gzip',
            ContentDisposition='attachment',
        )
        # free up memory
        response['Body'].close()
    else:
        # for csv, we can just modify the headers
        s3_client.copy_object(
            Bucket=bucket,
            Key=prefix,
            CopySource={'Bucket': bucket, 'Key': prefix},
            ContentType=content_type,
            ContentEncoding='gzip',
            ContentDisposition='attachment',
            MetadataDirective='REPLACE',
        )


def write_per_region(*, con: duckdb.DuckDBPyConnection, config: OCRConfig, region_type: RegionType):
    consolidated_buildings_uri = config.vector.building_geoparquet_uri
    per_region_output_prefix = config.vector.per_region_analysis_prefix

    if region_type == 'tract':
        region_ds = catalog.get_dataset('us-census-tracts')
    elif region_type == 'county':
        region_ds = catalog.get_dataset('us-census-counties')
    else:
        raise ValueError(f'region_type: {region_type} not supported. Must be: {RegionType}')

    region_path = UPath(f's3://{region_ds.bucket}/{region_ds.prefix}')

    if config.debug:
        console.log(f'Creating temporary table for {region_type} grouped risk data.')

    # create joined temp table
    con.execute(f"""
        CREATE TEMP TABLE {region_type}_grouped_risk AS
        SELECT
            b.NAME,
            b.GEOID,
            a.wind_risk_2011,
            a.wind_risk_2047,
            a.burn_probability_2011,
            a.burn_probability_2047,
            a.conditional_risk_usfs,
            a.burn_probability_usfs_2011,
            a.burn_probability_usfs_2047,
            ST_X(ST_Centroid(a.geometry)) AS centroid_longitude,
            ST_Y(ST_Centroid(a.geometry)) AS centroid_latitude,
            a.geometry
        FROM read_parquet('{consolidated_buildings_uri.as_uri()}') a
        JOIN read_parquet('{region_path}') b
            ON ST_Intersects(a.geometry, b.geometry)
    """)

    geoid_list = con.sql(
        f"""SELECT DISTINCT(GEOID) from {region_type}_grouped_risk"""
    ).fetchnumpy()['GEOID']

    if config.debug:
        console.log(f'Found {len(geoid_list)} {region_type}s to process.')

    # write csv
    csv_path = per_region_output_prefix / region_type / 'csv'
    csv_path.mkdir(parents=True, exist_ok=True)

    if config.debug:
        console.log(f'Writing CSV files to {csv_path}.')

    for geoid in track(geoid_list, description=f'Writing CSV files for {region_type}s'):
        ufname = csv_path / f'{geoid}.csv'
        con.execute(f"""COPY (
        SELECT
            * EXCLUDE (geometry, NAME, GEOID)
        FROM
            {region_type}_grouped_risk
        WHERE
            GEOID = '{geoid}'
        ) TO '{ufname.as_uri()}' (
        FORMAT CSV,
        COMPRESSION 'gzip',
        OVERWRITE_OR_IGNORE
        );""")

        if region_path.protocol == 's3':
            bucket = ufname.parts[0].strip('/')
            _modify_headers(
                bucket=bucket,
                prefix=ufname.path.split(bucket + '/')[1],
                content_type='text/csv',
            )

    if config.debug:
        console.log(f'Finished writing {len(geoid_list)} CSV files.')

    # write geojson
    geojson_path = per_region_output_prefix / region_type / 'geojson'
    geojson_path.mkdir(parents=True, exist_ok=True)

    if config.debug:
        console.log(f'Writing GeoJSON files to {geojson_path}.')

    for geoid in track(geoid_list, description=f'Writing GeoJSON files for {region_type}s'):
        ufname = geojson_path / f'{geoid}.geojson'
        con.execute(f"""COPY (
        SELECT
            * EXCLUDE (centroid_longitude, centroid_latitude, NAME, GEOID)
        FROM
            {region_type}_grouped_risk
        WHERE
            GEOID = '{geoid}'
        ) TO '{ufname.as_uri()}' (
        FORMAT GDAL,
        DRIVER 'GEOJSON',
        LAYER_NAME {region_type},
        OVERWRITE_OR_IGNORE
        );""")

        if region_path.protocol == 's3':
            bucket = ufname.parts[0].strip('/')
            _modify_headers(
                bucket=bucket,
                prefix=ufname.path.split(bucket + '/')[1],
                content_type='text/json',
            )

    if config.debug:
        console.log(f'Finished writing {len(geoid_list)} GeoJSON files.')


def write_per_region_analysis_files(config: OCRConfig):
    connection = duckdb.connect(database=':memory:')

    # Load required extensions (spatial + httpfs + aws) before any spatial ops or S3 reads
    install_load_extensions(aws=True, spatial=True, httpfs=True, con=connection)
    apply_s3_creds(con=connection)

    if config.debug:
        console.log('Writing per-region county files.')
    write_per_region(con=connection, config=config, region_type='county')
    if config.debug:
        console.log('Writing per-region tract files.')
    write_per_region(con=connection, config=config, region_type='tract')
