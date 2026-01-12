import subprocess
import tempfile
from pathlib import Path

import duckdb
from upath import UPath

from ocr.config import OCRConfig
from ocr.console import console
from ocr.utils import apply_s3_creds, copy_or_upload, get_temp_dir, install_load_extensions


def create_building_pmtiles(
    config: OCRConfig,
):
    """Convert consolidated geoparquet to PMTiles (using DuckDB Python API).

    Steps:
      1. (Optionally) create or reuse a DuckDB connection and load extensions.
      2. Export feature rows as NDJSON GeoJSON features via COPY ... TO.
      3. Invoke tippecanoe on the NDJSON to produce a PMTiles archive.
      4. Upload resulting PMTiles to the configured destination.
    """

    input_path = f'{config.vector.region_geoparquet_uri}/*.parquet'  # type: ignore[attr-defined]
    output_path = config.vector.buildings_pmtiles_uri  # type: ignore[attr-defined]

    needs_s3 = any(str(p).startswith('s3://') for p in [input_path, output_path])

    connection = duckdb.connect(database=':memory:')

    try:
        install_load_extensions(aws=needs_s3, spatial=True, httpfs=True, con=connection)
        if needs_s3:
            apply_s3_creds(region='us-west-2', con=connection)

        with tempfile.TemporaryDirectory(dir=get_temp_dir()) as tmpdir:
            tmp_path = UPath(tmpdir)
            local_pmtiles = tmp_path / 'aggregated.pmtiles'
            ndjson_path = Path(tmpdir) / 'buildings.ndjson'

            if config.debug:
                console.log(f'Exporting features from {input_path} to NDJSON in {ndjson_path}')

            copy_sql = f"""
            COPY (
                SELECT
                    'Feature' AS type,
                    json_object(
                        '0', wind_risk_2011,
                        '1', wind_risk_2047,
                        '2', burn_probability_2011,
                        '3', burn_probability_2047,
                        '4', conditional_risk_usfs,
                        '5', burn_probability_usfs_2011,
                        '6', burn_probability_usfs_2047
                    ) AS properties,
                    json(ST_AsGeoJson(geometry)) AS geometry
                FROM read_parquet('{input_path}')
            ) TO '{ndjson_path.as_posix()}' (FORMAT json);
            """
            connection.execute(copy_sql)

            if config.debug:
                console.log('NDJSON export complete')
                console.log(f'Generating PMTiles at {local_pmtiles}')

            tippecanoe_cmd = [
                'tippecanoe',
                '-o',
                str(local_pmtiles),
                '-l',
                'risk',
                '-n',
                'building',
                '-f',
                '-P',  # Parallel processing
                '--drop-smallest-as-needed',
                '-q',
                '--extend-zooms-if-still-dropping',
                '-zg',
                '-Z 6',
                '--generate-ids',
                str(ndjson_path),
            ]
            subprocess.run(tippecanoe_cmd, check=True)

            if config.debug:
                console.log('Tippecanoe tiles generation complete')
                console.log(f'Uploading PMTiles to {output_path}')

            copy_or_upload(local_pmtiles, output_path)

            if config.debug:
                console.log('PMTiles upload completed successfully')
    finally:
        try:
            connection.close()
        except Exception:
            pass
