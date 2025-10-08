"""For each (counties, census tract), write analysis files in multiple file formats"""

import gzip
from typing import Literal

import boto3
import duckdb
from upath import UPath

from ocr import catalog
from ocr.config import OCRConfig
from ocr.console import console
from ocr.types import RegionType
from ocr.utils import apply_s3_creds, install_load_extensions


def _modify_headers(bucket: str, prefix: str, content_type: Literal['text/csv', 'text/geojson']):
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
        )
    else:
        # for csv, we can just modify the headers
        s3_client.copy_object(
            Bucket=bucket,
            Key=prefix,
            CopySource={'Bucket': bucket, 'Key': prefix},
            ContentType=content_type,
            ContentEncoding='gzip',
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

    # create joined temp table
    con.execute(f""" CREATE TEMP TABLE {region_type}_grouped_risk AS
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
    ST_X (ST_Centroid (a.geometry)) AS longitude,
    ST_Y (ST_Centroid (a.geometry)) AS latitude,
    a.geometry
    from
    read_parquet('{consolidated_buildings_uri.as_uri()}') a
    JOIN read_parquet('{region_path}') b ON ST_Intersects (a.geometry, b.geometry)""")

    geoid_list = con.sql(
        f"""SELECT DISTINCT(GEOID) from {region_type}_grouped_risk"""
    ).fetchnumpy()['GEOID']

    # write csv
    csv_path = per_region_output_prefix / region_type / 'csv'
    csv_path.mkdir(parents=True, exist_ok=True)

    for geoid in geoid_list:
        fname = f'{csv_path}/{geoid}.csv'
        con.execute(f"""COPY (
        SELECT
            * EXCLUDE geometry
        FROM
            {region_type}_grouped_risk
        WHERE
            GEOID = '{geoid}'
        ) TO '{fname}' (
        FORMAT CSV,
        COMPRESSION 'gzip',
        OVERWRITE_OR_IGNORE
        );""")

    # write geojson
    geojson_path = per_region_output_prefix / region_type / 'geojson'
    geojson_path.mkdir(parents=True, exist_ok=True)
    for geoid in geoid_list:
        con.execute(f"""COPY (
        SELECT
            * EXCLUDE latitude,
            longitude
        FROM
            {region_type}_grouped_risk
        WHERE
            GEOID = '{geoid}'
        ) TO '{geojson_path}/{geoid}.geojson' (
        FORMAT GDAL,
        DRIVER 'GEOJSON',
        LAYER_NAME {region_type},
        OVERWRITE_OR_IGNORE
        );""")

    # Modify CSV and GeoJSON headers in s3 for data downloads
    if region_path.protocol == 's3':
        from cloudpathlib import AnyPath

        csv_path = AnyPath(csv_path)
        geojson_path = AnyPath(geojson_path)
        for geoid in geoid_list:
            _modify_headers(
                bucket=csv_path.bucket,
                prefix=(csv_path / f'{geoid}.csv').key,
                content_type='text/csv',
            )
            _modify_headers(
                bucket=geojson_path.bucket,
                prefix=(geojson_path / f'{geoid}.geojson').key,
                content_type='text/json',
            )


def write_per_region_analysis_files(config: OCRConfig):
    connection = duckdb.connect(database=':memory:')

    # Load required extensions (spatial + httpfs + aws) before any spatial ops or S3 reads
    install_load_extensions(aws=True, spatial=True, httpfs=True, con=connection)
    apply_s3_creds(con=connection)

    # NotImplementedException: Not implemented Error: GDAL Error (6): The GeoJSON driver does not overwrite existing files.
    # So we need to clear any existing files
    config.vector.delete_per_region_files()

    if config.debug:
        console.log('Writing per-region county files.')
    write_per_region(con=connection, config=config, region_type='county')
    if config.debug:
        console.log('Writing per-region tract files.')
    write_per_region(con=connection, config=config, region_type='tract')
