import os
import tempfile
from pathlib import Path

import dotenv
import typer

from ocr.config import OCRConfig
from ocr.deploy.managers import CoiledBatchManager, LocalBatchManager
from ocr.types import Platform, RiskType

app = typer.Typer(help='Run OCR deployment pipeline on Coiled')


def load_config(file_path: Path | None) -> OCRConfig:
    """
    Load OCR configuration from a YAML file.
    """

    if file_path is None:
        return OCRConfig()
    else:
        dotenv.load_dotenv(file_path)  # loads environment variables from the specified file
        return OCRConfig()  # loads from environment variables


@app.command()
def run(
    env_file: Path | None = typer.Option(
        None,
        '-e',
        '--env-file',
        help='Path to the environment variables file. These will be used to set up the OCRConfiguration',
        show_default=True,
        exists=True,
        file_okay=True,
        resolve_path=True,
    ),
    region_id: list[str] | None = typer.Option(
        None, '-r', '--region-id', help='Region IDs to process, e.g., y10_x2'
    ),
    all_region_ids: bool = typer.Option(
        False,
        '--all-region-ids',
        help='Process all valid region IDs',
        show_default=True,
    ),
    risk_type: RiskType = typer.Option(
        RiskType.FIRE, '-t', '--risk-type', help='Type of risk to calculate', show_default=True
    ),
    debug: bool = typer.Option(
        False, '-d', '--debug', help='Enable Debugging Mode', show_default=True
    ),
    summary_stats: bool = typer.Option(
        False,
        '-s',
        '--summary-stats',
        help='Adds in spatial summary aggregations.',
        show_default=True,
    ),
    platform: Platform = typer.Option(
        Platform.COILED,
        '-p',
        '--platform',
        help='Platform to run the pipeline on',
        show_default=True,
    ),
):
    """
    Run the OCR deployment pipeline on Coiled.
    """

    if all_region_ids and region_id:
        raise typer.BadParameter(
            'Cannot use --all-region-ids and -r/--region-id together. Please specify either one.'
        )
    if not all_region_ids and not region_id:
        raise typer.BadParameter('You must specify either --region-id or --all-region-ids.')

    from ocr.icechunk_utils import get_commit_messages_ancestry

    # ------------- CONFIG ---------------

    config = load_config(env_file)

    icechunk_repo_and_session = config.icechunk.repo_and_session()
    if all_region_ids:
        provided_region_ids = set(config.chunking.valid_region_ids)
    else:
        provided_region_ids = set(region_id or [])
    valid_region_ids = provided_region_ids.intersection(config.chunking.valid_region_ids)
    processed_region_ids = set(get_commit_messages_ancestry(icechunk_repo_and_session['repo']))
    unprocessed_valid_region_ids = valid_region_ids.difference(processed_region_ids)

    if len(unprocessed_valid_region_ids) == 0:
        invalid_region_ids = provided_region_ids.difference(config.chunking.valid_region_ids)
        previously_processed_ids = provided_region_ids.intersection(processed_region_ids)
        error_message = 'No valid region IDs to process. All provided region IDs were rejected for the following reasons:\n'

        if invalid_region_ids:
            error_message += f'- Invalid region IDs: {", ".join(sorted(invalid_region_ids))}\n'
            error_message += f'  Valid region IDs: {", ".join(sorted(list(config.chunking.valid_region_ids)))}...\n'

        if previously_processed_ids:
            error_message += (
                f'- Already processed region IDs: {", ".join(sorted(previously_processed_ids))}\n'
            )

        error_message += "\nPlease provide valid region IDs that haven't been processed yet."

        raise ValueError(error_message)

    if platform == Platform.COILED:
        shared_coiled_kwargs = {
            'ntasks': 1,
            'region': 'us-west-2',
            'forward_aws_credentials': True,
            'tag': {'Project': 'OCR'},
        }

        env_vars = {}
        if env_file is not None:
            env_vars = dotenv.dotenv_values(str(env_file))

        # ------------- 01 AU ---------------

        batch_manager_01 = CoiledBatchManager(debug=debug)

        for rid in unprocessed_valid_region_ids:
            batch_manager_01.submit_job(
                command=f'ocr process-region {rid} --risk-type {risk_type.value}',
                name=f'process-region-{rid}-{config.branch.value}',
                kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.large', 'env': env_vars},
            )

        # # this is a monitoring / blocking func. We should be able to block with this, then run 02, 03 etc.
        batch_manager_01.wait_for_completion()

        # ----------- 02 Aggregate -------------
        batch_manager_aggregate_02 = CoiledBatchManager(debug=debug)
        batch_manager_aggregate_02.submit_job(
            command='ocr aggregate',
            name=f'aggregate-geoparquet-{config.branch.value}',
            kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.large', 'env': env_vars},
        )
        batch_manager_aggregate_02.wait_for_completion()

        if summary_stats:
            batch_manager_county_aggregation_01 = CoiledBatchManager(debug=debug)
            batch_manager_county_aggregation_01.submit_job(
                command='ocr aggregate-regional-risk',
                name=f'create-aggregated-region-summary-stats-{config.branch.value}',
                kwargs={
                    **shared_coiled_kwargs,
                    'vm_type': 'm8g.2xlarge',
                    'env': env_vars,
                },
            )
            batch_manager_county_aggregation_01.wait_for_completion()

            # create summary stats PMTiles layer
            batch_manager_county_tiles_02 = CoiledBatchManager(debug=debug)
            batch_manager_county_tiles_02.submit_job(
                command='ocr create-regional-pmtiles',
                name=f'create-aggregated-region-pmtiles-{config.branch.value}',
                kwargs={
                    **shared_coiled_kwargs,
                    'vm_type': 'c7a.2xlarge',
                    'env': env_vars,
                },
            )

        # ------------- 03  Tiles ---------------
        batch_manager_03 = CoiledBatchManager(debug=debug)
        batch_manager_03.submit_job(
            command='ocr create-pmtiles',
            name=f'create-pmtiles-{config.branch.value}',
            kwargs={
                **shared_coiled_kwargs,
                'vm_type': 'c7a.xlarge',
                'env': env_vars,
            },
        )

        batch_manager_03.wait_for_completion()

    elif platform == Platform.LOCAL:
        manager = LocalBatchManager(debug=debug)
        env = os.environ.copy()
        tmp_dir = tempfile.gettempdir()

        for rid in unprocessed_valid_region_ids:
            manager.submit_job(
                command=f'ocr process-region {rid} --risk-type {risk_type.value}',
                name=f'process-region-{rid}-{config.branch.value}',
                kwargs={
                    'env': env,
                    'cwd': tmp_dir,
                },
            )
        manager.wait_for_completion()

        # Aggregate geoparquet regions
        manager = LocalBatchManager(debug=debug)
        manager.submit_job(
            command='ocr aggregate',
            name=f'aggregate-geoparquet-{config.branch.value}',
            kwargs={
                'env': env,
                'cwd': tmp_dir,
            },
        )
        manager.wait_for_completion()
        if summary_stats:
            manager = LocalBatchManager(debug=debug)
            # Aggregate regional fire and wind risk statistics
            manager.submit_job(
                command='ocr aggregate-regional-risk',
                name=f'create-aggregated-region-summary-stats-{config.branch.value}',
                kwargs={
                    'env': env,
                    'cwd': tmp_dir,
                },
            )
            manager.wait_for_completion()

            # Create summary stats PMTiles layer
            manager = LocalBatchManager(debug=debug)
            manager.submit_job(
                command='ocr create-regional-pmtiles',
                name=f'create-aggregated-region-pmtiles-{config.branch.value}',
                kwargs={
                    'env': env,
                    'cwd': tmp_dir,
                },
            )
            manager.wait_for_completion()
        # Create PMTiles from the consolidated geoparquet file
        manager = LocalBatchManager(debug=debug)
        manager.submit_job(
            command='ocr create-pmtiles',
            name=f'create-pmtiles-{config.branch.value}',
            kwargs={
                'env': env,
                'cwd': tmp_dir,
            },
        )
        manager.wait_for_completion()


@app.command()
def process_region(
    env_file: Path | None = typer.Option(
        None,
        '-e',
        '--env-file',
        help='Path to the environment variables file. These will be used to set up the OCRConfiguration',
        show_default=True,
        exists=True,
        file_okay=True,
        resolve_path=True,
    ),
    region_id: str = typer.Argument(..., help='Region ID to process, e.g., y10_x2'),
    risk_type: RiskType = typer.Option(
        RiskType.FIRE, '-t', '--risk-type', help='Type of risk to calculate', show_default=True
    ),
):
    """
    Calculate and write risk for a given region to Icechunk CONUS template.
    """

    from ocr.pipeline.process_region import calculate_risk

    config = load_config(env_file)

    calculate_risk(config=config, region_id=region_id, risk_type=risk_type)


@app.command()
def aggregate(
    env_file: Path | None = typer.Option(
        None,
        '-e',
        '--env-file',
        help='Path to the environment variables file. These will be used to set up the OCRConfiguration',
        show_default=True,
        exists=True,
        file_okay=True,
        resolve_path=True,
    ),
):
    """
    Aggregate geoparquet regions, reproject and write.
    """
    from ocr.pipeline.aggregate import aggregated_gpq

    config = load_config(env_file)
    aggregated_gpq(
        input_path=config.vector.region_geoparquet_uri,
        output_path=config.vector.consolidated_geoparquet_uri,
    )


@app.command()
def aggregate_regional_risk(
    env_file: Path | None = typer.Option(
        None,
        '-e',
        '--env-file',
        help='Path to the environment variables file. These will be used to set up the OCRConfiguration',
        show_default=True,
        exists=True,
        file_okay=True,
        resolve_path=True,
    ),
):
    """
    Aggregate regional fire and wind risk statistics.
    """
    from ocr.pipeline.fire_wind_risk_regional_aggregator import (
        compute_regional_fire_wind_risk_statistics,
    )

    config = load_config(env_file)

    compute_regional_fire_wind_risk_statistics(
        counties_path=config.vector.counties_geoparquet_uri,
        tracts_path=config.vector.tracts_geoparquet_uri,
        consolidated_buildings_path=config.vector.consolidated_geoparquet_uri,
        aggregated_regions_prefix=config.vector.aggregated_regions_prefix,
    )


@app.command()
def create_regional_pmtiles(
    env_file: Path | None = typer.Option(
        None,
        '-e',
        '--env-file',
        help='Path to the environment variables file. These will be used to set up the OCRConfiguration',
        show_default=True,
        exists=True,
        file_okay=True,
        resolve_path=True,
    ),
):
    """
    Create PMTiles for regional risk statistics (counties and tracts).
    """
    from ocr.pipeline.create_regional_pmtiles import create_regional_pmtiles

    config = load_config(env_file)

    create_regional_pmtiles(
        tract_stats_path=config.vector.tracts_geoparquet_uri,
        county_stats_path=config.vector.counties_geoparquet_uri,
        tract_pmtiles_output=config.vector.region_geoparquet_uri / 'tract.pmtiles',
        county_pmtiles_output=config.vector.region_geoparquet_uri / 'counties.pmtiles',
    )


@app.command()
def create_pmtiles(
    env_file: Path | None = typer.Option(
        None,
        '-e',
        '--env-file',
        help='Path to the environment variables file. These will be used to set up the OCRConfiguration',
        show_default=True,
        exists=True,
        file_okay=True,
        resolve_path=True,
    ),
):
    """
    Create PMTiles from the consolidated geoparquet file.
    """
    from ocr.pipeline.create_pmtiles import create_pmtiles

    config = load_config(env_file)

    create_pmtiles(
        input_path=config.vector.consolidated_geoparquet_uri,
        output_path=config.vector.pmtiles_prefix_uri,
    )


if __name__ == '__main__':
    typer.run(run)
