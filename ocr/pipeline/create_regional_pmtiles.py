import subprocess
import tempfile
from pathlib import Path

from ocr.console import console
from ocr.types import Branch


def create_regional_pmtiles(branch: Branch):
    """
    Create PMTiles for tract and county regional risk statistics.

    This function runs DuckDB queries on regional statistics, creates PMTiles using tippecanoe,
    and uploads the results to S3.
    """
    branch_value = branch.value
    s3_base = 's3://carbonplan-ocr'
    intermediate_base = f'{s3_base}/intermediate/fire-risk/vector/{branch_value}'

    tract_stats_path = f'{intermediate_base}/region_aggregation/tract/tract_summary_stats.parquet'
    county_stats_path = (
        f'{intermediate_base}/region_aggregation/county/county_summary_stats.parquet'
    )

    tract_pmtiles_output = f'{intermediate_base}/region_aggregation/tract/tract.pmtiles'
    county_pmtiles_output = f'{intermediate_base}/region_aggregation/county/counties.pmtiles'

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tract_pmtiles = tmp_path / 'tract.pmtiles'
        county_pmtiles = tmp_path / 'counties.pmtiles'

        console.log(f'Creating tract PMTiles from {tract_stats_path}')
        duckdb_tract_query = f"""
        load spatial;

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

            FROM read_parquet('{tract_stats_path}')
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

        tippecanoe_proc = subprocess.run(tippecanoe_cmd, stdin=duckdb_proc.stdout, check=True)

        console.log('Tract PMTiles created successfully')

        console.log(f'Creating county PMTiles from {county_stats_path}')
        duckdb_county_query = f"""
        load spatial;

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

            FROM read_parquet('{county_stats_path}')
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

        tippecanoe_proc = subprocess.run(tippecanoe_cmd, stdin=duckdb_proc.stdout, check=True)

        console.log('County PMTiles created successfully')

        console.log(f'Uploading tract PMTiles to {tract_pmtiles_output}')
        subprocess.run(
            ['s5cmd', 'cp', '--sp', str(tract_pmtiles), tract_pmtiles_output], check=True
        )

        console.log(f'Uploading county PMTiles to {county_pmtiles_output}')
        subprocess.run(
            ['s5cmd', 'cp', '--sp', str(county_pmtiles), county_pmtiles_output], check=True
        )

        console.log('PMTiles uploads completed successfully')
