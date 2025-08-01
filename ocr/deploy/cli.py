import os
import tempfile
from pathlib import Path

import typer
import yaml

from ocr.config import OCRConfig
from ocr.console import console
from ocr.deploy.managers import CoiledBatchManager, LocalBatchManager
from ocr.types import Platform, RiskType

app = typer.Typer(help='Run OCR deployment pipeline on Coiled')


def load_config_from_yaml(file_path: Path) -> OCRConfig:
    """
    Load OCR configuration from a YAML file.
    """
    text = file_path.read_text()
    config_data = yaml.safe_load(text)
    return OCRConfig.model_validate(config_data)


@app.command()
def run(
    config: Path = typer.Option(
        'ocr-config.local.yaml',
        '-c',
        '--config',
        help='Path to the OCR configuration file in YAML format',
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

    config_ = load_config_from_yaml(config)
    if debug:
        console.log(f'Using OCR config: {config_}')

    icechunk_repo_and_session = config_.icechunk.repo_and_session()
    if all_region_ids:
        provided_region_ids = set(config_.chunking.valid_region_ids)
    else:
        provided_region_ids = set(region_id or [])
    valid_region_ids = provided_region_ids.intersection(config_.chunking.valid_region_ids)
    processed_region_ids = set(get_commit_messages_ancestry(icechunk_repo_and_session['repo']))
    unprocessed_valid_region_ids = valid_region_ids.difference(processed_region_ids)

    if len(unprocessed_valid_region_ids) == 0:
        invalid_region_ids = provided_region_ids.difference(config_.chunking.valid_region_ids)
        previously_processed_ids = provided_region_ids.intersection(processed_region_ids)
        error_message = 'No valid region IDs to process. All provided region IDs were rejected for the following reasons:\n'

        if invalid_region_ids:
            error_message += f'- Invalid region IDs: {", ".join(sorted(invalid_region_ids))}\n'
            error_message += f'  Valid region IDs: {", ".join(sorted(list(config_.chunking.valid_region_ids)))}...\n'

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
                command=f'ocr process-region {rid} --config {config.name} --risk-type {risk_type.value}',
                name=f'process-region-{rid}-{config_.branch.value}',
                kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.large', 'file': str(config)},
            )

        # # this is a monitoring / blocking func. We should be able to block with this, then run 02, 03 etc.
        batch_manager_01.wait_for_completion()

        # ----------- 02 Aggregate -------------
        batch_manager_aggregate_02 = CoiledBatchManager(debug=debug)
        batch_manager_aggregate_02.submit_job(
            command=f'ocr aggregate --config {config.name}',
            name=f'aggregate-geoparquet-{config_.branch.value}',
            kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.large', 'file': str(config)},
        )
        batch_manager_aggregate_02.wait_for_completion()

        if summary_stats:
            batch_manager_county_aggregation_01 = CoiledBatchManager(debug=debug)
            batch_manager_county_aggregation_01.submit_job(
                command=f'ocr aggregate-regional-risk --config {config.name}',
                name=f'create-aggregated-region-summary-stats-{config_.branch.value}',
                kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.2xlarge', 'file': str(config)},
            )
            batch_manager_county_aggregation_01.wait_for_completion()

            # create summary stats PMTiles layer
            batch_manager_county_tiles_02 = CoiledBatchManager(debug=debug)
            batch_manager_county_tiles_02.submit_job(
                command=f'ocr create-regional-pmtiles --config {config.name}',
                name=f'create-aggregated-region-pmtiles-{config_.branch.value}',
                kwargs={
                    **shared_coiled_kwargs,
                    'vm_type': 'c7a.2xlarge',
                    'file': str(config),
                },
            )

        # ------------- 03  Tiles ---------------
        batch_manager_03 = CoiledBatchManager(debug=debug)
        batch_manager_03.submit_job(
            command=f'ocr create-pmtiles --config {config.name}',
            name=f'create-pmtiles-{config_.branch.value}',
            kwargs={
                **shared_coiled_kwargs,
                'vm_type': 'c7a.xlarge',
                'file': str(config),
            },
        )

        batch_manager_03.wait_for_completion()

    elif platform == Platform.LOCAL:
        manager = LocalBatchManager(debug=debug)
        env = os.environ.copy()
        tmp_dir = tempfile.gettempdir()

        for rid in unprocessed_valid_region_ids:
            manager.submit_job(
                command=f'ocr process-region {rid} --config {config} --risk-type {risk_type.value}',
                name=f'process-region-{rid}-{config_.branch.value}',
                kwargs={
                    'env': env,
                    'cwd': tmp_dir,
                },
            )
        manager.wait_for_completion()

        # Aggregate geoparquet regions
        manager = LocalBatchManager(debug=debug)
        manager.submit_job(
            command=f'ocr aggregate --config {config}',
            name=f'aggregate-geoparquet-{config_.branch.value}',
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
                command=f'ocr aggregate-regional-risk --config {config}',
                name=f'create-aggregated-region-summary-stats-{config_.branch.value}',
                kwargs={
                    'env': env,
                    'cwd': tmp_dir,
                },
            )
            manager.wait_for_completion()

            # Create summary stats PMTiles layer
            manager = LocalBatchManager(debug=debug)
            manager.submit_job(
                command=f'ocr create-regional-pmtiles --config {config}',
                name=f'create-aggregated-region-pmtiles-{config_.branch.value}',
                kwargs={
                    'env': env,
                    'cwd': tmp_dir,
                },
            )
            manager.wait_for_completion()
        # Create PMTiles from the consolidated geoparquet file
        manager = LocalBatchManager(debug=debug)
        manager.submit_job(
            command=f'ocr create-pmtiles --config {config}',
            name=f'create-pmtiles-{config_.branch.value}',
            kwargs={
                'env': env,
                'cwd': tmp_dir,
            },
        )
        manager.wait_for_completion()


@app.command()
def process_region(
    region_id: str = typer.Argument(..., help='Region ID to process, e.g., y10_x2'),
    config: Path = typer.Option(
        'ocr-config.local.yaml',
        '-c',
        '--config',
        help='Path to the OCR configuration file in YAML format',
        show_default=True,
        exists=True,
        file_okay=True,
    ),
    risk_type: RiskType = typer.Option(
        RiskType.FIRE, '-t', '--risk-type', help='Type of risk to calculate', show_default=True
    ),
):
    """
    Calculate and write risk for a given region to Icechunk CONUS template.
    """

    from ocr.pipeline.process_region import calculate_risk

    config_ = load_config_from_yaml(config)

    calculate_risk(config=config_, region_id=region_id, risk_type=risk_type)


@app.command()
def aggregate(
    config: Path = typer.Option(
        'ocr-config.local.yaml',
        '-c',
        '--config',
        help='Path to the OCR configuration file in YAML format',
        show_default=True,
        exists=True,
        file_okay=True,
    ),
):
    """
    Aggregate geoparquet regions, reproject and write.
    """
    from ocr.pipeline.aggregate import aggregated_gpq

    config_ = load_config_from_yaml(config)
    aggregated_gpq(
        input_path=config_.vector.region_geoparquet_uri,
        output_path=config_.vector.consolidated_geoparquet_uri,
    )


@app.command()
def aggregate_regional_risk(
    config: Path = typer.Option(
        'ocr-config.local.yaml',
        '-c',
        '--config',
        help='Path to the OCR configuration file in YAML format',
        show_default=True,
        exists=True,
        file_okay=True,
    ),
):
    """
    Aggregate regional fire and wind risk statistics.
    """
    from ocr.pipeline.fire_wind_risk_regional_aggregator import (
        compute_regional_fire_wind_risk_statistics,
    )

    config_ = load_config_from_yaml(config)

    compute_regional_fire_wind_risk_statistics(
        counties_path=config_.vector.counties_geoparquet_uri,
        tracts_path=config_.vector.tracts_geoparquet_uri,
        consolidated_buildings_path=config_.vector.consolidated_geoparquet_uri,
        aggregated_regions_prefix=config_.vector.aggregated_regions_prefix,
    )


@app.command()
def create_regional_pmtiles(
    config: Path = typer.Option(
        'ocr-config.local.yaml',
        '-c',
        '--config',
        help='Path to the OCR configuration file in YAML format',
        show_default=True,
        exists=True,
        file_okay=True,
    ),
):
    """
    Create PMTiles for regional risk statistics (counties and tracts).
    """
    from ocr.pipeline.create_regional_pmtiles import create_regional_pmtiles

    config_ = load_config_from_yaml(config)

    create_regional_pmtiles(
        tract_stats_path=config_.vector.tracts_geoparquet_uri,
        county_stats_path=config_.vector.counties_geoparquet_uri,
        tract_pmtiles_output=config_.vector.region_geoparquet_uri / 'tract.pmtiles',
        county_pmtiles_output=config_.vector.region_geoparquet_uri / 'counties.pmtiles',
    )


@app.command()
def create_pmtiles(
    config: Path = typer.Option(
        'ocr-config.local.yaml',
        '-c',
        '--config',
        help='Path to the OCR configuration file in YAML format',
        show_default=True,
        exists=True,
        file_okay=True,
    ),
):
    """
    Create PMTiles from the consolidated geoparquet file.
    """
    from ocr.pipeline.create_pmtiles import create_pmtiles

    config_ = load_config_from_yaml(config)

    create_pmtiles(
        input_path=config_.vector.consolidated_geoparquet_uri,
        output_path=config_.vector.pmtiles_prefix_uri,
    )


if __name__ == '__main__':
    typer.run(run)
