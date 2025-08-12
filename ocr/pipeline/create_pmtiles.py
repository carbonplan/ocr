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
    2. Reads the geoparquet with duckdb spatial
    3. Creates PMTiles using tippecanoe
    4. Uploads the result back to S3
    """

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
                    'risk_2011', risk_2011,
                    'risk_2047', risk_2047,
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
        ]

        _ = subprocess.run(tippecanoe_cmd, stdin=duckdb_proc.stdout, check=True)

        console.log('Tippecanoe tiles generation complete')

        console.log(f'Uploading PMTiles to {output_path}')

        copy_or_upload(local_pmtiles, output_path)

        console.log('PMTiles upload completed successfully')
