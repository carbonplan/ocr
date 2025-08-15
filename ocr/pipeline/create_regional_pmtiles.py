import subprocess
import tempfile

from upath import UPath

from ocr.config import OCRConfig
from ocr.console import console


def create_regional_pmtiles(
    config: OCRConfig,
):
    """
    Create PMTiles for tract and county regional risk statistics.

    This function runs DuckDB queries on regional statistics, creates PMTiles using tippecanoe,
    and uploads the results to S3.
    """

    tracts_summary_stats_path = config.vector.tracts_summary_stats_uri
    counties_summary_stats_path = config.vector.counties_summary_stats_uri
    tract_pmtiles_output = config.vector.tracts_pmtiles_uri
    county_pmtiles_output = config.vector.counties_pmtiles_uri

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = UPath(tmpdir)
        tract_pmtiles = tmp_path / 'tract.pmtiles'
        county_pmtiles = tmp_path / 'counties.pmtiles'

        console.log(f'Creating tract PMTiles from {tracts_summary_stats_path}')
        duckdb_tract_query = f"""
        install spatial; load spatial; install httpfs; load httpfs;

        COPY (
            SELECT
                'Feature' AS type,
                json_object(
                    'tract_geoid', NAME,
                    'building_count', building_count,
                    'avg_risk_2011_horizon_1', avg_risk_2011_horizon_1,
                    'avg_risk_2011_horizon_15', avg_risk_2011_horizon_15,
                    'avg_risk_2011_horizon_30', avg_risk_2011_horizon_30,
                    'avg_risk_2047_horizon_1', avg_risk_2047_horizon_1,
                    'avg_risk_2047_horizon_15', avg_risk_2047_horizon_15,
                    'avg_risk_2047_horizon_30', avg_risk_2047_horizon_30,
                    'avg_wind_risk_2011_horizon_1', avg_wind_risk_2011_horizon_1,
                    'avg_wind_risk_2011_horizon_15', avg_wind_risk_2011_horizon_15,
                    'avg_wind_risk_2011_horizon_30', avg_wind_risk_2011_horizon_30,
                    'avg_wind_risk_2047_horizon_1', avg_wind_risk_2047_horizon_1,
                    'avg_wind_risk_2047_horizon_15', avg_wind_risk_2047_horizon_15,
                    'avg_wind_risk_2047_horizon_15', avg_wind_risk_2047_horizon_15,
                    'risk_2011_horizon_1', risk_2011_horizon_1,
                    'risk_2011_horizon_15', risk_2011_horizon_15,
                    'risk_2011_horizon_30', risk_2011_horizon_30,
                    'risk_2047_horizon_1', risk_2047_horizon_1,
                    'risk_2047_horizon_15', risk_2047_horizon_15,
                    'risk_2047_horizon_30', risk_2047_horizon_30,
                    'wind_risk_2011_horizon_1', wind_risk_2011_horizon_1,
                    'wind_risk_2011_horizon_15', wind_risk_2011_horizon_15,
                    'wind_risk_2011_horizon_30', wind_risk_2011_horizon_30,
                    'wind_risk_2047_horizon_1', wind_risk_2047_horizon_1,
                    'wind_risk_2047_horizon_15', wind_risk_2047_horizon_15,
                    'wind_risk_2047_horizon_30', wind_risk_2047_horizon_30
                     ) AS properties,
                json(ST_AsGeoJson(geometry)) AS geometry

            FROM read_parquet('{tracts_summary_stats_path}')
        ) TO STDOUT (FORMAT json);
        """

        # Run duckdb to generate GeoJSON and pipe to tippecanoe
        duckdb_proc = subprocess.Popen(['duckdb', '-c', duckdb_tract_query], stdout=subprocess.PIPE)

        tippecanoe_cmd = [
            'tippecanoe',
            '-o',
            str(tract_pmtiles),
            '-l',
            'risk',
            '-n',
            'tract',
            '-f',
            '-P',
            '--drop-smallest-as-needed',
            '-q',
            '--extend-zooms-if-still-dropping',
            '-zg',
        ]

        _ = subprocess.run(tippecanoe_cmd, stdin=duckdb_proc.stdout, check=True)

        console.log('Tract PMTiles created successfully')

        console.log(f'Creating county PMTiles from {counties_summary_stats_path}')
        duckdb_county_query = f"""
        install spatial; load spatial; install httpfs; load httpfs;
        COPY (
            SELECT
                'Feature' AS type,
                json_object(
                    'county_name', NAME,
                    'building_count', building_count,
                    'avg_risk_2011_horizon_1', avg_risk_2011_horizon_1,
                    'avg_risk_2011_horizon_15', avg_risk_2011_horizon_15,
                    'avg_risk_2011_horizon_30', avg_risk_2011_horizon_30,
                    'avg_risk_2047_horizon_1', avg_risk_2047_horizon_1,
                    'avg_risk_2047_horizon_15', avg_risk_2047_horizon_15,
                    'avg_risk_2047_horizon_30', avg_risk_2047_horizon_30,
                    'avg_wind_risk_2011_horizon_1', avg_wind_risk_2011_horizon_1,
                    'avg_wind_risk_2011_horizon_15', avg_wind_risk_2011_horizon_15,
                    'avg_wind_risk_2011_horizon_30', avg_wind_risk_2011_horizon_30,
                    'avg_wind_risk_2047_horizon_1', avg_wind_risk_2047_horizon_1,
                    'avg_wind_risk_2047_horizon_15', avg_wind_risk_2047_horizon_15,
                    'avg_wind_risk_2047_horizon_30', avg_wind_risk_2047_horizon_30,
                    'risk_2011_horizon_1', risk_2011_horizon_1,
                    'risk_2011_horizon_15', risk_2011_horizon_15,
                    'risk_2011_horizon_30', risk_2011_horizon_30,
                    'risk_2047_horizon_1', risk_2047_horizon_1,
                    'risk_2047_horizon_15', risk_2047_horizon_15,
                    'risk_2047_horizon_30', risk_2047_horizon_30,
                    'wind_risk_2011_horizon_1', wind_risk_2011_horizon_1,
                    'wind_risk_2011_horizon_15', wind_risk_2011_horizon_15,
                    'wind_risk_2011_horizon_30', wind_risk_2011_horizon_30,
                    'wind_risk_2047_horizon_1', wind_risk_2047_horizon_1,
                    'wind_risk_2047_horizon_15', wind_risk_2047_horizon_15,
                    'wind_risk_2047_horizon_30', wind_risk_2047_horizon_30
                     ) AS properties,
                json(ST_AsGeoJson(geometry)) AS geometry

            FROM read_parquet('{counties_summary_stats_path}')
        ) TO STDOUT (FORMAT json);
        """

        duckdb_proc = subprocess.Popen(
            ['duckdb', '-c', duckdb_county_query], stdout=subprocess.PIPE
        )

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
        ]

        _ = subprocess.run(tippecanoe_cmd, stdin=duckdb_proc.stdout, check=True)

        console.log('County PMTiles created successfully')

        def copy_or_upload(src: UPath, dest: UPath):
            import shutil

            if dest.protocol == 's3':
                subprocess.run(['s5cmd', 'cp', '--sp', str(src), str(dest)], check=True)
            else:
                shutil.copy(str(src), str(dest))

        console.log(f'Uploading tract PMTiles to {tract_pmtiles_output}')
        copy_or_upload(tract_pmtiles, tract_pmtiles_output)

        console.log(f'Uploading county PMTiles to {county_pmtiles_output}')
        copy_or_upload(county_pmtiles, county_pmtiles_output)

        console.log('PMTiles uploads completed successfully')
