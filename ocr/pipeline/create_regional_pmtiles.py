import subprocess
import tempfile
from pathlib import Path

import duckdb
from upath import UPath

from ocr.config import OCRConfig
from ocr.console import console
from ocr.utils import apply_s3_creds, copy_or_upload, install_load_extensions


def create_regional_pmtiles(
    config: OCRConfig,
):
    """Create PMTiles for tract and county regional risk statistics.

    Refactored to use DuckDB's Python API instead of spawning the DuckDB CLI.
    We materialize the feature rows to newline-delimited GeoJSON (NDJSON) files
    via DuckDB ``COPY ... TO`` and then invoke ``tippecanoe`` on those files.

    Parameters
    ----------
    config : OCRConfig
        Project configuration object (provides input/output URIs and debug flag).

    """

    # Access vector config attributes (these exist on OCRConfig.vector)
    tracts_summary_stats_path = config.vector.tracts_summary_stats_uri  # type: ignore[attr-defined]
    counties_summary_stats_path = config.vector.counties_summary_stats_uri  # type: ignore[attr-defined]
    tract_pmtiles_output = config.vector.tracts_pmtiles_uri  # type: ignore[attr-defined]
    county_pmtiles_output = config.vector.counties_pmtiles_uri  # type: ignore[attr-defined]

    # Determine if we need S3 credentials (basic heuristic: any input/output on s3)
    needs_s3 = any(
        str(p).startswith('s3://') for p in [tracts_summary_stats_path, counties_summary_stats_path]
    )

    connection = duckdb.connect(database=':memory:')

    try:
        # Install/load required extensions
        install_load_extensions(aws=needs_s3, spatial=True, httpfs=True, con=connection)
        if needs_s3:
            # Apply credentials so httpfs/aws can access objects; region optional (default us-west-2)
            apply_s3_creds(region='us-west-2', con=connection)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = UPath(tmpdir)
            tract_pmtiles = tmp_path / 'tract.pmtiles'
            county_pmtiles = tmp_path / 'counties.pmtiles'
            tract_ndjson = Path(tmpdir) / 'tract.ndjson'
            county_ndjson = Path(tmpdir) / 'county.ndjson'

            if config.debug:
                console.log(f'Creating tract PMTiles from {tracts_summary_stats_path}')
            duckdb_tract_copy = f"""
            COPY (
                SELECT
                    'Feature' AS type,
                    json_object(
                        'tract_geoid', NAME,
                        'building_count', building_count,
                        'avg_USFS_RPS_horizon_1', avg_USFS_RPS_horizon_1,
                        'avg_USFS_RPS_horizon_15', avg_USFS_RPS_horizon_15,
                        'avg_USFS_RPS_horizon_30', avg_USFS_RPS_horizon_30,
                        'avg_wind_risk_2011_horizon_1', avg_wind_risk_2011_horizon_1,
                        'avg_wind_risk_2011_horizon_15', avg_wind_risk_2011_horizon_15,
                        'avg_wind_risk_2011_horizon_30', avg_wind_risk_2011_horizon_30,
                        'avg_wind_risk_2047_horizon_1', avg_wind_risk_2047_horizon_1,
                        'avg_wind_risk_2047_horizon_15', avg_wind_risk_2047_horizon_15,
                        'avg_wind_risk_2047_horizon_30', avg_wind_risk_2047_horizon_30,
                        'USFS_RPS_horizon_1', USFS_RPS_horizon_1,
                        'USFS_RPS_horizon_15', USFS_RPS_horizon_15,
                        'USFS_RPS_horizon_30', USFS_RPS_horizon_30,
                        'wind_risk_2011_horizon_1', wind_risk_2011_horizon_1,
                        'wind_risk_2011_horizon_15', wind_risk_2011_horizon_15,
                        'wind_risk_2011_horizon_30', wind_risk_2011_horizon_30,
                        'wind_risk_2047_horizon_1', wind_risk_2047_horizon_1,
                        'wind_risk_2047_horizon_15', wind_risk_2047_horizon_15,
                        'wind_risk_2047_horizon_30', wind_risk_2047_horizon_30
                    ) AS properties,
                    json(ST_AsGeoJson(geometry)) AS geometry
                FROM read_parquet('{tracts_summary_stats_path}')
            ) TO '{tract_ndjson.as_posix()}' (FORMAT json);
            """
            connection.execute(duckdb_tract_copy)

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
                console.log(f'Creating county PMTiles from {counties_summary_stats_path}')
            duckdb_county_copy = f"""
            COPY (
                SELECT
                    'Feature' AS type,
                    json_object(
                        'county_name', NAME,
                        'building_count', building_count,
                        'avg_USFS_RPS_horizon_1', avg_USFS_RPS_horizon_1,
                        'avg_USFS_RPS_horizon_15', avg_USFS_RPS_horizon_15,
                        'avg_USFS_RPS_horizon_30', avg_USFS_RPS_horizon_30,
                        'avg_wind_risk_2011_horizon_1', avg_wind_risk_2011_horizon_1,
                        'avg_wind_risk_2011_horizon_15', avg_wind_risk_2011_horizon_15,
                        'avg_wind_risk_2011_horizon_30', avg_wind_risk_2011_horizon_30,
                        'avg_wind_risk_2047_horizon_1', avg_wind_risk_2047_horizon_1,
                        'avg_wind_risk_2047_horizon_15', avg_wind_risk_2047_horizon_15,
                        'avg_wind_risk_2047_horizon_30', avg_wind_risk_2047_horizon_30,
                        'USFS_RPS_horizon_1', USFS_RPS_horizon_1,
                        'USFS_RPS1_horizon_15', USFS_RPS_horizon_15,
                        'USFS_RPS_horizon_30', USFS_RPS_horizon_30,
                        'wind_risk_2011_horizon_1', wind_risk_2011_horizon_1,
                        'wind_risk_2011_horizon_15', wind_risk_2011_horizon_15,
                        'wind_risk_2011_horizon_30', wind_risk_2011_horizon_30,
                        'wind_risk_2047_horizon_1', wind_risk_2047_horizon_1,
                        'wind_risk_2047_horizon_15', wind_risk_2047_horizon_15,
                        'wind_risk_2047_horizon_30', wind_risk_2047_horizon_30
                    ) AS properties,
                    json(ST_AsGeoJson(geometry)) AS geometry
                FROM read_parquet('{counties_summary_stats_path}')
            ) TO '{county_ndjson.as_posix()}' (FORMAT json);
            """
            connection.execute(duckdb_county_copy)

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
