import os
import tempfile

import typer

from ocr.deploy.managers import CoiledBatchManager, LocalBatchManager
from ocr.types import Branch, Platform, RiskType

app = typer.Typer(help='Run OCR deployment pipeline on Coiled')


@app.command()
def run(
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
        RiskType.WIND, '-t', '--risk-type', help='Type of risk to calculate', show_default=True
    ),
    branch: Branch = typer.Option(
        'QA', '-b', '--branch', help='Data branch path', show_default=True
    ),
    wipe: bool = typer.Option(
        False,
        '-w',
        '--wipe',
        help='If True, wipes icechunk repo and vector data before initializing.',
        show_default=True,
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

    from ocr.chunking_config import ChunkingConfig
    from ocr.icechunk_utils import IcechunkConfig, VectorConfig, get_commit_messages_ancestry

    # ------------- CONFIG ---------------
    branch_ = branch.value
    config = ChunkingConfig()
    VectorConfig(branch=branch_, wipe=wipe)
    icechunk_config = IcechunkConfig(branch=branch_, wipe=wipe)

    icechunk_repo_and_session = icechunk_config.repo_and_session()
    if all_region_ids:
        provided_region_ids = set(config.valid_region_ids)
    else:
        provided_region_ids = set(region_id or [])
    valid_region_ids = provided_region_ids.intersection(config.valid_region_ids)
    processed_region_ids = set(get_commit_messages_ancestry(icechunk_repo_and_session['repo']))
    unprocessed_valid_region_ids = valid_region_ids.difference(processed_region_ids)

    if len(unprocessed_valid_region_ids) == 0:
        invalid_region_ids = provided_region_ids.difference(config.valid_region_ids)
        previously_processed_ids = provided_region_ids.intersection(processed_region_ids)
        error_message = 'No valid region IDs to process. All provided region IDs were rejected for the following reasons:\n'

        if invalid_region_ids:
            error_message += f'- Invalid region IDs: {", ".join(sorted(invalid_region_ids))}\n'
            error_message += (
                f'  Valid region IDs: {", ".join(sorted(list(config.valid_region_ids)))}...\n'
            )

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

        # ------------- 01 AU ---------------

        batch_manager_01 = CoiledBatchManager(debug=debug)

        for rid in unprocessed_valid_region_ids:
            batch_manager_01.submit_job(
                command=f'ocr process-region {rid} --branch {branch.value} --risk-type {risk_type.value}',
                name=f'process-region-{rid}-{branch.value}',
                kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.large'},
            )

        # # this is a monitoring / blocking func. We should be able to block with this, then run 02, 03 etc.
        batch_manager_01.wait_for_completion()

        # ----------- 02 Aggregate -------------
        batch_manager_aggregate_02 = CoiledBatchManager(debug=debug)
        batch_manager_aggregate_02.submit_job(
            command=f'ocr aggregate --branch {branch.value}',
            name=f'aggregate-geoparquet-{branch.value}',
            kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.large'},
        )
        batch_manager_aggregate_02.wait_for_completion()

        if summary_stats:
            batch_manager_county_aggregation_01 = CoiledBatchManager(debug=debug)
            batch_manager_county_aggregation_01.submit_job(
                command=f'ocr aggregate-regional-risk --branch {branch.value}',
                name=f'create-aggregated-region-summary-stats-{branch.value}',
                kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.2xlarge'},
            )
            batch_manager_county_aggregation_01.wait_for_completion()

            # create summary stats PMTiles layer
            batch_manager_county_tiles_02 = CoiledBatchManager(debug=debug)
            batch_manager_county_tiles_02.submit_job(
                command=f'ocr create-regional-pmtiles --branch {branch.value}',
                name=f'create-aggregated-region-pmtiles-{branch.value}',
                kwargs={
                    **shared_coiled_kwargs,
                    'vm_type': 'c7a.2xlarge',
                    'container': 'quay.io/carbonplan/ocr:latest',
                },
            )

        # ------------- 03  Tiles ---------------
        batch_manager_03 = CoiledBatchManager(debug=debug)
        batch_manager_03.submit_job(
            command=f'ocr create-pmtiles --branch {branch.value}',
            name=f'create-pmtiles-{branch.value}',
            kwargs={
                **shared_coiled_kwargs,
                'vm_type': 'c7a.xlarge',
            },
        )

        batch_manager_03.wait_for_completion()

    elif platform == Platform.LOCAL:
        manager = LocalBatchManager(debug=debug)
        env = os.environ.copy()
        tmp_dir = tempfile.gettempdir()

        for rid in unprocessed_valid_region_ids:
            manager.submit_job(
                command=f'ocr process-region {rid} --branch {branch.value} --risk-type {risk_type.value}',
                name=f'process-region-{rid}-{branch.value}',
                kwargs={
                    'env': env,
                    'cwd': tmp_dir,
                },
            )
        manager.wait_for_completion()

        # Aggregate geoparquet regions
        manager = LocalBatchManager(debug=debug)
        manager.submit_job(
            command=f'ocr aggregate --branch {branch.value}',
            name=f'aggregate-geoparquet-{branch.value}',
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
                command=f'ocr aggregate-regional-risk --branch {branch.value}',
                name=f'create-aggregated-region-summary-stats-{branch.value}',
                kwargs={
                    'env': env,
                    'cwd': tmp_dir,
                },
            )
            manager.wait_for_completion()

            # Create summary stats PMTiles layer
            manager = LocalBatchManager(debug=debug)
            manager.submit_job(
                command=f'ocr create-regional-pmtiles --branch {branch.value}',
                name=f'create-aggregated-region-pmtiles-{branch.value}',
                kwargs={
                    'env': env,
                    'cwd': tmp_dir,
                },
            )
            manager.wait_for_completion()
        # Create PMTiles from the consolidated geoparquet file
        manager = LocalBatchManager(debug=debug)
        manager.submit_job(
            command=f'ocr create-pmtiles --branch {branch.value}',
            name=f'create-pmtiles-{branch.value}',
            kwargs={
                'env': env,
                'cwd': tmp_dir,
            },
        )
        manager.wait_for_completion()


@app.command()
def process_region(
    region_id: str = typer.Argument(..., help='Region ID to process, e.g., y10_x2'),
    risk_type: RiskType = typer.Option(
        RiskType.WIND, '-t', '--risk-type', help='Type of risk to calculate', show_default=True
    ),
    branch: Branch = typer.Option(
        'QA', '-b', '--branch', help='Data branch path', show_default=True
    ),
    wipe: bool = typer.Option(
        False,
        '-w',
        '--wipe',
        help='If True, wipes icechunk repo and vector data before initializing.',
        show_default=True,
    ),
):
    """
    Calculate and write risk for a given region to Icechunk CONUS template.
    """
    from ocr.config import OCRConfig
    from ocr.pipeline.process_region import calculate_risk

    config = OCRConfig(storage_root='/tmp', branch=branch, wipe=wipe)

    calculate_risk(config=config, region_id=region_id, risk_type=risk_type)


@app.command()
def aggregate(
    branch: Branch = typer.Option(
        'QA', '-b', '--branch', help='Data branch path', show_default=True
    ),
):
    """
    Aggregate geoparquet regions, reproject and write.
    """
    from ocr.pipeline.aggregate import aggregated_gpq

    aggregated_gpq(branch=branch)


@app.command()
def aggregate_regional_risk(
    branch: Branch = typer.Option(
        'QA', '-b', '--branch', help='Data branch path', show_default=True
    ),
):
    """
    Aggregate regional fire and wind risk statistics.
    """
    from ocr.pipeline.fire_wind_risk_regional_aggregator import (
        compute_regional_fire_wind_risk_statistics,
    )

    compute_regional_fire_wind_risk_statistics(branch=branch)


@app.command()
def create_regional_pmtiles(
    branch: Branch = typer.Option(
        'QA', '-b', '--branch', help='Data branch path', show_default=True
    ),
):
    """
    Create PMTiles for regional risk statistics (counties and tracts).
    """
    from ocr.pipeline.create_regional_pmtiles import create_regional_pmtiles

    create_regional_pmtiles(branch=branch)


@app.command()
def create_pmtiles(
    branch: Branch = typer.Option(
        'QA', '-b', '--branch', help='Data branch path', show_default=True
    ),
):
    """
    Create PMTiles from the consolidated geoparquet file.
    """
    from ocr.pipeline.create_pmtiles import create_pmtiles

    create_pmtiles(branch=branch)


if __name__ == '__main__':
    typer.run(run)
