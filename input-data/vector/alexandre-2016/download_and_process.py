import pathlib

import boto3
import geopandas as gpd
import pandas as pd
import patoolib
import requests
from shapely.geometry import Point

bucket = 'carbonplan-ocr'
prefix = 'input/fire-risk/vector/alexandre-2016'


def get_download_url() -> str:
    # Base URL for the Dryad API
    base_url = 'https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.h1v2g'

    # Fetch dataset metadata
    response = requests.get(base_url)
    dataset_info = response.json()

    # Extract the download link for the dataset
    download_link = dataset_info['_links']['stash:download']['href']

    # Construct the full download URL
    download_url = f'https://datadryad.org{download_link}'

    print(f'Download URL: {download_url}')

    return download_url


def download_to_s3(url: str):
    output_file = '/tmp/USBurnedBuildings.rar'
    if pathlib.Path(output_file).exists():
        print(f'{output_file} already exists, skipping download.')

    else:
        response = requests.get(url, stream=True)
        with open(output_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f'Downloaded {output_file}')

    # Extract the contents of the RAR file
    patoolib.extract_archive(output_file, outdir='/tmp/USBurnedBuildings')
    print(f'Extracted {output_file} to /tmp/USBurnedBuildings')

    # Upload the extracted files to S3
    s3_client = boto3.client('s3', region_name='us-west-2')

    for file in pathlib.Path('/tmp/USBurnedBuildings').iterdir():
        if file.is_file():
            s3_key = f'{prefix}/{file.name}'
            # if the file already exists in S3, skip it
            try:
                s3_client.head_object(Bucket=bucket, Key=s3_key)
                print(f'{s3_key} already exists in S3, skipping upload.')
            except Exception as _:
                print(f'{s3_key} does not exist in S3, uploading...')
                s3_client.upload_file(str(file), bucket, s3_key)
                print(f'Uploaded {file} to s3://{bucket}/{s3_key}')


def process_digitized_buildings():
    filename = 'digitized_buildings_2000_2010.csv'
    key = f'{prefix}/{filename}'
    url = f's3://{bucket}/{key}'
    out_filename = 'digitized_buildings_2000_2010.parquet'
    out_key = f'{prefix}/{out_filename}'
    output_url = f's3://{bucket}/{out_key}'
    # check if the file exists in S3
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket, Key=out_key)
        print(f'{filename} already exists in S3: {output_url}, skipping processing.')
        return
    except Exception as _:
        print(f'{filename} does not exist in S3: {output_url}, processing...')
        try:
            df = pd.read_csv(url)
            print(f'Loaded {filename} from S3.')
            geometry = [Point(xy) for xy in zip(df['X'], df['Y'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:5070')  # EPSG:4269 is NAD83
            gdf.to_parquet(output_url, engine='pyarrow', compression='zstd')
            print(f'Converted {filename} to Parquet and uploaded to S3: {output_url}.')
        except Exception as e:
            print(f'Error processing {filename} from S3: {e}')


if __name__ == '__main__':
    url = get_download_url()
    download_to_s3(url)
    process_digitized_buildings()
