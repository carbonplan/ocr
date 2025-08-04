import subprocess
import tempfile
from pathlib import Path

from ocr.console import console
from ocr.types import Branch


def create_pmtiles(branch: Branch):
    branch_value = branch.value
    s3_base = 's3://carbonplan-ocr'

    input_path = (
        f'{s3_base}/intermediate/fire-risk/vector/{branch_value}/consolidated_geoparquet.parquet'
    )
    output_path = f'{s3_base}/intermediate/fire-risk/vector/{branch_value}/aggregated.pmtiles'

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        local_pmtiles = tmp_path / 'aggregated.pmtiles'

        console.log(f'Creating building PMTiles from {input_path}')
        duckdb_tract_query = f"""
        load spatial;
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

        # Run duckdb to generate GeoJSON and pipe to tippecanoe
        duckdb_proc = subprocess.Popen(['duckdb', '-c', duckdb_tract_query], stdout=subprocess.PIPE)

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
        subprocess.run(['s5cmd', 'cp', '--sp', str(local_pmtiles), output_path], check=True)

        console.log('PMTiles upload completed successfully')
