import subprocess
import tempfile

from upath import UPath

from ocr.console import console


def copy_or_upload(src: UPath, dest: UPath):
    import shutil

    if dest.protocol == 's3' or src.protocol == 's3':
        subprocess.run(['s5cmd', 'cp', '--sp', str(src), str(dest)], check=True)
    else:
        shutil.copy(str(src), str(dest))


def create_pmtiles(*, input_path: UPath, output_path: UPath):
    """
    Convert consolidated geoparquet to PMTiles format.

    This function:
    1. Downloads the consolidated geoparquet file from S3
    2. Converts it to FlatGeobuf format using ogr2ogr
    3. Creates PMTiles using tippecanoe
    4. Uploads the result back to S3
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = UPath(tmpdir)
        local_parquet = tmp_path / 'region.parquet'
        local_fgb = tmp_path / 'region.fgb'
        local_pmtiles = tmp_path / 'aggregated.pmtiles'

        console.log(f'Downloading consolidated geoparquet from {input_path}')
        copy_or_upload(input_path, local_parquet)

        console.log('Converting to FlatGeobuf format')
        subprocess.run(
            [
                'ogr2ogr',
                '-progress',
                '-f',
                'FlatGeobuf',
                str(local_fgb),
                str(local_parquet),
                '-nlt',
                'PROMOTE_TO_MULTI',
            ],
            check=True,
        )

        console.log('GDAL conversion complete')

        console.log('Generating PMTiles using tippecanoe')
        subprocess.run(
            [
                'tippecanoe',
                '-o',
                str(local_pmtiles),
                '-l',
                'risk',
                '-n',
                'USFS BP Risk',
                '-f',
                '-P',
                '--drop-smallest-as-needed',
                '-q',
                '--extend-zooms-if-still-dropping',
                '-zg',
                str(local_fgb),
            ],
            check=True,
        )

        console.log('Tippecanoe tiles generation complete')

        console.log(f'Uploading PMTiles to {output_path}')

        copy_or_upload(local_pmtiles, output_path)

        console.log('PMTiles upload completed successfully')
