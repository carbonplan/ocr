import subprocess
import tempfile

from upath import UPath

from ocr.config import OCRConfig
from ocr.console import console
from ocr.utils import copy_or_upload


def create_pmtiles(config: OCRConfig):
    """
    Convert consolidated geoparquet to PMTiles format.

    This function:
    2. Reads the geoparquet with duckdb spatial
    3. Creates PMTiles using tippecanoe
    4. Uploads the result back to S3
    """

    input_path = config.vector.building_geoparquet_uri
    output_path = config.vector.buildings_pmtiles_uri

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = UPath(tmpdir)
        local_pmtiles = tmp_path / 'aggregated.pmtiles'

        # Run duckdb to generate GeoJSON and pipe to tippecanoe
        duckdb_building_query = f"""
        install spatial; load spatial; install httpfs; load httpfs;
        COPY (
            SELECT
                'Feature' AS type,
                json_object(
                    'USFS_RPS', USFS_RPS,
                    'wind_risk_2011', wind_risk_2011,
                    'wind_risk_2047', wind_risk_2047
                     ) AS properties,
                json(ST_AsGeoJson(geometry)) AS geometry
            FROM read_parquet('{input_path}')
        ) TO STDOUT (FORMAT json);
        """
        duckdb_proc = subprocess.Popen(
            ['duckdb', '-c', duckdb_building_query], stdout=subprocess.PIPE
        )

        tippecanoe_cmd = [
            'tippecanoe',
            '-o',
            str(local_pmtiles),
            '-l',
            'risk',
            '-n',
            'building',
            '-f',
            '-P',
            '--drop-smallest-as-needed',
            '-q',
            '--extend-zooms-if-still-dropping',
            '-zg',
            '--generate-ids',
        ]

        _ = subprocess.run(tippecanoe_cmd, stdin=duckdb_proc.stdout, check=True)

        if config.debug:
            console.log('Tippecanoe tiles generation complete')
            console.log(f'Uploading PMTiles to {output_path}')

        copy_or_upload(local_pmtiles, output_path)

        if config.debug:
            console.log('PMTiles upload completed successfully')
