import subprocess
import tempfile
from pathlib import Path

import duckdb
from upath import UPath

from ocr.config import OCRConfig
from ocr.console import console
from ocr.utils import apply_s3_creds, copy_or_upload, get_temp_dir, install_load_extensions


def create_building_centroid_pmtiles(
    config: OCRConfig,
):
    """Nearly identical to create_building_pmtiles.py, but creates centroid only layer for higher zoom levels."""

    input_path = f'{config.vector.region_geoparquet_uri}/*.parquet'  # type: ignore[attr-defined]

    output_path = config.vector.building_centroids_pmtiles_uri  # type: ignore[attr-defined]

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
                    '0', ROUND(CAST(wind_risk_2011 AS DOUBLE), 3),
                    '1', ROUND(CAST(wind_risk_2047 AS DOUBLE), 3)
                    ) AS properties,
                    json(ST_AsGeoJson(ST_Centroid(geometry))) AS geometry
                FROM read_parquet('{input_path}')
                WHERE
                    wind_risk_2011 > 0
                    AND
                    wind_risk_2047 > 0
            ) TO '{ndjson_path.as_posix()}' (FORMAT json);
            """
            connection.execute(copy_sql)

            if config.debug:
                console.log('NDJSON export complete')
                console.log(f'Generating PMTiles at {local_pmtiles}')
            # import ipdb; ipdb.set_trace()
            tippecanoe_cmd = [
                'tippecanoe',
                '-o',
                str(local_pmtiles),
                '-l',
                'risk',
                '-n',
                'centroid',
                '-f',
                '-P',
                '--drop-fraction-as-needed',
                '--maximum-tile-features=400000',
                '--maximum-tile-bytes=1000000',
                '-z',
                '13',
                '-q',
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
