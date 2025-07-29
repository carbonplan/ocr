import subprocess
import tempfile
from pathlib import Path

from ocr.console import console
from ocr.types import Branch


def create_pmtiles(branch: Branch):
    """
    Convert consolidated geoparquet to PMTiles format.

    This function:
    1. Downloads the consolidated geoparquet file from S3
    2. Converts it to FlatGeobuf format using ogr2ogr
    3. Creates PMTiles using tippecanoe
    4. Uploads the result back to S3
    """
    branch_value = branch.value
    s3_base = 's3://carbonplan-ocr'

    input_path = (
        f'{s3_base}/intermediate/fire-risk/vector/{branch_value}/consolidated_geoparquet.parquet'
    )
    output_path = f'{s3_base}/intermediate/fire-risk/vector/{branch_value}/aggregated.pmtiles'

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        local_parquet = tmp_path / 'region.parquet'
        local_fgb = tmp_path / 'region.fgb'
        local_pmtiles = tmp_path / 'aggregated.pmtiles'

        console.log(f'Downloading consolidated geoparquet from {input_path}')
        subprocess.run(['s5cmd', 'cp', '--sp', input_path, str(local_parquet)], check=True)

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
        subprocess.run(['s5cmd', 'cp', '--sp', str(local_pmtiles), output_path], check=True)

        console.log('PMTiles upload completed successfully')
