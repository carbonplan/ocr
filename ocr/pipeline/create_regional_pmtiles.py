import subprocess
import tempfile
from pathlib import Path

import duckdb
from upath import UPath

from ocr.config import OCRConfig
from ocr.console import console
from ocr.utils import apply_s3_creds, copy_or_upload, get_temp_dir, install_load_extensions


def create_regional_pmtiles(
    config: OCRConfig,
):
    """Create PMTiles for county, tract and block regional risk statistics.

    Refactored to use DuckDB's Python API instead of spawning the DuckDB CLI.
    We materialize the feature rows to newline-delimited GeoJSON (NDJSON) files
    via DuckDB ``COPY ... TO`` and then invoke ``tippecanoe`` on those files.

    Parameters
    ----------
    config : OCRConfig
        Project configuration object (provides input/output URIs and debug flag).

    """

    # Access vector config attributes (these exist on OCRConfig.vector)
    block_summary_stats_path = config.vector.block_summary_stats_uri  # type: ignore[attr-defined]
    tracts_summary_stats_path = config.vector.block_summary_stats_uri  # type: ignore[attr-defined]
    counties_summary_stats_path = config.vector.counties_summary_stats_uri  # type: ignore[attr-defined]

    block_pmtiles_output = config.vector.block_pmtiles_uri  # type: ignore[attr-defined]
    tract_pmtiles_output = config.vector.tracts_pmtiles_uri  # type: ignore[attr-defined]
    county_pmtiles_output = config.vector.counties_pmtiles_uri  # type: ignore[attr-defined]

    # Determine if we need S3 credentials (basic heuristic: any input/output on s3)
    needs_s3 = any(
        str(p).startswith('s3://')
        for p in [block_summary_stats_path, tracts_summary_stats_path, counties_summary_stats_path]
    )

    connection = duckdb.connect(database=':memory:')

    try:
        # Install/load required extensions
        install_load_extensions(aws=needs_s3, spatial=True, httpfs=True, con=connection)
        if needs_s3:
            # Apply credentials so httpfs/aws can access objects; region optional (default us-west-2)
            apply_s3_creds(region='us-west-2', con=connection)

        with tempfile.TemporaryDirectory(dir=get_temp_dir()) as tmpdir:
            tmp_path = UPath(tmpdir)
            block_pmtiles = tmp_path / 'blocks.pmtiles'
            tract_pmtiles = tmp_path / 'tracts.pmtiles'
            county_pmtiles = tmp_path / 'counties.pmtiles'

            block_ndjson = Path(tmpdir) / 'blocks.ndjson'
            tract_ndjson = Path(tmpdir) / 'tracts.ndjson'
            county_ndjson = Path(tmpdir) / 'counties.ndjson'

            if config.debug:
                console.log(f'Creating block PMTiles from {block_summary_stats_path}')

            if config.debug:
                console.log(
                    f'Exporting NDJSON features: {block_ndjson} from {block_summary_stats_path} for block PMTiles'
                )
            duckdb_block_copy = f"""
            COPY (
                SELECT
                    'Feature' AS type,
                    json_object(
                        '0', building_count,
                        '1', mean_wind_risk_2011,
                        '2', mean_wind_risk_2047,
                        '3', median_wind_risk_2011,
                        '4', median_wind_risk_2047,
                        '5', wind_risk_2011,
                        '6', wind_risk_2047,
                        '7', GEOID,
                        '8', [
                                ST_XMin(geometry),
                                ST_YMin(geometry),
                                ST_XMax(geometry),
                                ST_YMax(geometry)
                            ]

                    ) AS properties,
                    json(ST_AsGeoJson(geometry)) AS geometry
                FROM read_parquet('{block_summary_stats_path}')
            ) TO '{block_ndjson.as_posix()}' (FORMAT json);
            """
            connection.execute(duckdb_block_copy)

            if config.debug:
                console.log(f'Generating block PMTiles at {block_pmtiles}')

            tippecanoe_cmd = [
                'tippecanoe',
                '-o',
                str(block_pmtiles),
                '-l',
                'risk',
                '-n',
                'block',
                '-f',
                '-P',  # input is newline-delimited features
                '--drop-smallest-as-needed',
                '-q',
                '--extend-zooms-if-still-dropping',
                '-zg',
                str(block_ndjson),
            ]
            subprocess.run(tippecanoe_cmd, check=True)
            if config.debug:
                console.log('block PMTiles created successfully')

            if config.debug:
                console.log(
                    f'Exporting tract PMTiles from {tracts_summary_stats_path} to {tract_ndjson}'
                )
            duckdb_tract_copy = f"""
            COPY (
                SELECT
                    'Feature' AS type,
                    json_object(
                        '0', building_count,
                        '1', mean_wind_risk_2011,
                        '2', mean_wind_risk_2047,
                        '3', median_wind_risk_2011,
                        '4', median_wind_risk_2047,
                        '5', wind_risk_2011,
                        '6', wind_risk_2047,
                        '7', GEOID,
                        '8', [
                                ST_XMin(geometry),
                                ST_YMin(geometry),
                                ST_XMax(geometry),
                                ST_YMax(geometry)
                            ]

                    ) AS properties,
                    json(ST_AsGeoJson(geometry)) AS geometry
                FROM read_parquet('{tracts_summary_stats_path}')
            ) TO '{tract_ndjson.as_posix()}' (FORMAT json);
            """
            connection.execute(duckdb_tract_copy)

            if config.debug:
                console.log(f'Generating tract PMTiles at {tract_pmtiles} from {tract_ndjson}')

            tippecanoe_cmd = [
                'tippecanoe',
                '-o',
                str(tract_pmtiles),
                '-l',
                'risk',
                '-n',
                'tract',
                '-f',
                '-P',  # input is newline-delimited features
                '--drop-smallest-as-needed',
                '-q',
                '--extend-zooms-if-still-dropping',
                '-zg',
                str(tract_ndjson),
            ]
            subprocess.run(tippecanoe_cmd, check=True)
            if config.debug:
                console.log('Tract PMTiles created successfully')

            if config.debug:
                console.log(
                    f'Exporting county PMTiles from {counties_summary_stats_path} to {county_ndjson}'
                )
            duckdb_county_copy = f"""
            COPY (
                SELECT
                    'Feature' AS type,
                    json_object(

                        '0', building_count,
                        '1', mean_wind_risk_2011,
                        '2', mean_wind_risk_2047,
                        '3', median_wind_risk_2011,
                        '4', median_wind_risk_2047,
                        '5', wind_risk_2011,
                        '6', wind_risk_2047,
                        '7', GEOID,
                        '8', [
                                ST_XMin(geometry),
                                ST_YMin(geometry),
                                ST_XMax(geometry),
                                ST_YMax(geometry)
                            ],
                        '9', NAME,
                    ) AS properties,
                    json(ST_AsGeoJson(geometry)) AS geometry
                FROM read_parquet('{counties_summary_stats_path}')
            ) TO '{county_ndjson.as_posix()}' (FORMAT json);
            """
            connection.execute(duckdb_county_copy)
            if config.debug:
                console.log(f'Generating county PMTiles at {county_pmtiles} from {county_ndjson}')

            tippecanoe_cmd = [
                'tippecanoe',
                '-o',
                str(county_pmtiles),
                '-l',
                'risk',
                '-n',
                'county',
                '-f',
                '-P',
                '--drop-smallest-as-needed',
                '-q',
                '--extend-zooms-if-still-dropping',
                '-zg',
                str(county_ndjson),
            ]
            subprocess.run(tippecanoe_cmd, check=True)
            if config.debug:
                console.log('County PMTiles created successfully')

            if config.debug:
                console.log(f'Uploading block PMTiles to {block_pmtiles_output}')
            copy_or_upload(block_pmtiles, block_pmtiles_output)

            if config.debug:
                console.log(f'Uploading tract PMTiles to {tract_pmtiles_output}')
            copy_or_upload(tract_pmtiles, tract_pmtiles_output)

            if config.debug:
                console.log(f'Uploading county PMTiles to {county_pmtiles_output}')
            copy_or_upload(county_pmtiles, county_pmtiles_output)

            if config.debug:
                console.log('PMTiles uploads completed successfully')
    finally:
        try:
            connection.close()
        except Exception:
            pass
