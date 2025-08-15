import os
import tempfile
import typing
from pathlib import Path

import dotenv
import typer

from ocr.config import OCRConfig
from ocr.deploy.managers import CoiledBatchManager, LocalBatchManager
from ocr.types import Platform, RiskType

app = typer.Typer(help='Run OCR deployment pipeline on Coiled')


def _resolve_env_vars(env_file: Path | None) -> dict[str, str]:
    if env_file is None:
        return {}
    # dotenv_values returns a dict[str, str | None] — coerce to str-only
    values = dotenv.dotenv_values(str(env_file)) or {}
    return {k: v for k, v in values.items() if v is not None}


def _coiled_kwargs(config: OCRConfig, env_file: Path | None) -> dict[str, typing.Any]:
    env_vars = _resolve_env_vars(env_file)
    env_vars = {**env_vars, 'BATCH_ENV_FLAG': '1'}
    return {**config.coiled.model_dump(), 'env': env_vars}


def _local_kwargs() -> dict[str, typing.Any]:
    env = os.environ.copy()
    # Ensure the in-batch flag is present to prevent re-submission loops
    env['BATCH_ENV_FLAG'] = '1'
    tmp_dir = tempfile.gettempdir()
    return {'env': env, 'cwd': tmp_dir}


def _get_manager(platform: Platform, debug: bool):
    if platform == Platform.COILED:
        return CoiledBatchManager(debug=debug)
    elif platform == Platform.LOCAL:
        return LocalBatchManager(debug=debug)
    else:
        raise ValueError(f'Unknown platform: {platform}')


def _in_batch() -> bool:
    return os.environ.get('BATCH_ENV_FLAG') == '1'


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
        Platform.LOCAL,
        '-p',
        '--platform',
        help='Platform to run the pipeline on',
        show_default=True,
    ),
    wipe: bool = typer.Option(
        False,
        '--wipe',
        help='Wipe the icechunk and vector data storages before running the pipeline',
        show_default=True,
    ),
    dispatch_platform: Platform | None = typer.Option(
        None,
        '--dispatch-platform',
        help='If set, schedule this run command on the specified platform instead of running inline.',
        show_default=True,
    ),
    vm_type: str | None = typer.Option(
        None, '--vm-type', help='VM type override for dispatch-platform (Coiled only).'
    ),
):
    """
    Run the OCR deployment pipeline. This will process regions, aggregate geoparquet files,
    and create PMTiles layers for the specified risk type.
    """

    if dispatch_platform is not None and not _in_batch():
        # rebuild the command to execute inside the dispatched job
        parts: list[str] = ['ocr run']
        if region_id:
            for rid in region_id:
                parts += ['-r', rid]
        if all_region_ids:
            parts += ['--all-region-ids']
        parts += ['--risk-type', risk_type.value]
        parts += ['--platform', platform.value]
        if summary_stats:
            parts += ['--summary-stats']
        if wipe:
            parts += ['--wipe']
        if debug:
            parts += ['--debug']
        # forward --env-file
        parts += ['--env-file', str(env_file)] if env_file else []

        command = ' '.join(parts)

        config = load_config(env_file)
        manager = _get_manager(dispatch_platform, debug)
        name = f'run-{config.branch.value}'

        if dispatch_platform == Platform.COILED:
            kwargs = {**_coiled_kwargs(config, env_file)}
            if vm_type:
                kwargs['vm_type'] = vm_type
        else:
            kwargs = {**_local_kwargs()}

        manager.submit_job(command=command, name=name, kwargs=kwargs)
        return

    if all_region_ids and region_id:
        raise typer.BadParameter(
            'Cannot use --all-region-ids and -r/--region-id together. Please specify either one.'
        )
    if not all_region_ids and not region_id:
        raise typer.BadParameter('You must specify either --region-id or --all-region-ids.')

    from ocr.icechunk_utils import get_commit_messages_ancestry

    # ------------- CONFIG ---------------

    config = load_config(env_file)
    if wipe:
        config.icechunk.wipe()
        config.vector.wipe()

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
        # ------------- 01 AU ---------------

        batch_manager_01 = CoiledBatchManager(debug=debug)

        # Submit one mapped job instead of one job per region
        kwargs = _coiled_kwargs(config, env_file)
        del kwargs['ntasks']
        batch_manager_01.submit_job(
            command=f'ocr process-region $COILED_BATCH_TASK_INPUT --risk-type {risk_type.value}',
            name=f'process-region-{config.branch.value}',
            kwargs={
                **kwargs,
                'map_over_values': sorted(list(unprocessed_valid_region_ids)),
            },
        )

        # # this is a monitoring / blocking func. We should be able to block with this, then run 02, 03 etc.
        batch_manager_01.wait_for_completion(exit_on_failure=True)

        # ----------- 02 Aggregate -------------
        batch_manager_aggregate_02 = CoiledBatchManager(debug=debug)
        batch_manager_aggregate_02.submit_job(
            command='ocr aggregate',
            name=f'aggregate-geoparquet-{config.branch.value}',
            kwargs={
                **_coiled_kwargs(config, env_file),
            },
        )
        batch_manager_aggregate_02.wait_for_completion(exit_on_failure=True)

        if summary_stats:
            batch_manager_county_aggregation_01 = CoiledBatchManager(debug=debug)
            batch_manager_county_aggregation_01.submit_job(
                command='ocr aggregate-region-risk-summary-stats',
                name=f'create-aggregated-region-summary-stats-{config.branch.value}',
                kwargs={
                    **_coiled_kwargs(config, env_file),
                    'vm_type': 'm8g.2xlarge',
                },
            )
            batch_manager_county_aggregation_01.wait_for_completion(exit_on_failure=True)

            # create summary stats PMTiles layer
            batch_manager_county_tiles_02 = CoiledBatchManager(debug=debug)
            batch_manager_county_tiles_02.submit_job(
                command='ocr create-regional-pmtiles',
                name=f'create-aggregated-region-pmtiles-{config.branch.value}',
                kwargs={
                    **_coiled_kwargs(config, env_file),
                    'vm_type': 'c8g.8xlarge',
                },
            )

        # ------------- 03  Tiles ---------------

        batch_manager_03 = CoiledBatchManager(debug=debug)
        batch_manager_03.submit_job(
            command='ocr create-pmtiles',
            name=f'create-pmtiles-{config.branch.value}',
            kwargs={
                **_coiled_kwargs(config, env_file),
                'vm_type': 'c8g.8xlarge',
            },
        )

        batch_manager_03.wait_for_completion(exit_on_failure=True)

    elif platform == Platform.LOCAL:
        manager = LocalBatchManager(debug=debug)

        for rid in unprocessed_valid_region_ids:
            manager.submit_job(
                command=f'ocr process-region {rid} --risk-type {risk_type.value}',
                name=f'process-region-{rid}-{config.branch.value}',
                kwargs={
                    **_local_kwargs(),
                },
            )
        manager.wait_for_completion(exit_on_failure=True)

        # Aggregate geoparquet regions
        manager = LocalBatchManager(debug=debug)
        manager.submit_job(
            command='ocr aggregate',
            name=f'aggregate-geoparquet-{config.branch.value}',
            kwargs={
                **_local_kwargs(),
            },
        )
        manager.wait_for_completion(exit_on_failure=True)

        if summary_stats:
            manager = LocalBatchManager(debug=debug)
            # Aggregate regional fire and wind risk statistics
            manager.submit_job(
                command='ocr aggregate-region-risk-summary-stats',
                name=f'create-aggregated-region-summary-stats-{config.branch.value}',
                kwargs={
                    **_local_kwargs(),
                },
            )
            manager.wait_for_completion(exit_on_failure=True)

            # Create summary stats PMTiles layer
            manager = LocalBatchManager(debug=debug)
            manager.submit_job(
                command='ocr create-regional-pmtiles',
                name=f'create-aggregated-region-pmtiles-{config.branch.value}',
                kwargs={
                    **_local_kwargs(),
                },
            )
            manager.wait_for_completion(exit_on_failure=True)

        # Create PMTiles from the consolidated geoparquet file
        manager = LocalBatchManager(debug=debug)
        manager.submit_job(
            command='ocr create-pmtiles',
            name=f'create-pmtiles-{config.branch.value}',
            kwargs={
                **_local_kwargs(),
            },
        )
        manager.wait_for_completion(exit_on_failure=True)


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
    platform: Platform | None = typer.Option(
        None,
        '-p',
        '--platform',
        help='If set, schedule this command on the specified platform instead of running inline.',
        show_default=True,
    ),
    debug: bool = typer.Option(
        False,
        '-d',
        '--debug',
        help='Enable Debugging Mode for the batch manager',
        show_default=True,
    ),
    vm_type: str | None = typer.Option(
        None, '--vm-type', help='Coiled VM type override (Coiled only).'
    ),
):
    """
    Calculate and write risk for a given region to Icechunk CONUS template.
    """

    # Schedule if requested and not already inside a batch task
    if platform is not None and not _in_batch():
        config = load_config(env_file)
        manager = _get_manager(platform, debug)
        command = f'ocr process-region {region_id} --risk-type {risk_type.value}'
        name = f'process-region-{region_id}-{config.branch.value}'

        if platform == Platform.COILED:
            kwargs = {**_coiled_kwargs(config, env_file)}
            if vm_type:
                kwargs['vm_type'] = vm_type
        else:
            kwargs = {**_local_kwargs()}

        manager.submit_job(command=command, name=name, kwargs=kwargs)
        manager.wait_for_completion(exit_on_failure=True)
        return

    from ocr.pipeline.process_region import calculate_risk

    config = load_config(env_file)

    y_slice, x_slice = config.chunking.region_id_to_latlon_slices(region_id=region_id)

    calculate_risk(
        region_geoparquet_uri=config.vector.region_geoparquet_uri,
        region_id=region_id,
        y_slice=y_slice,
        x_slice=x_slice,
        risk_type=risk_type,
        session=config.icechunk.repo_and_session()['session'],
    )


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
    platform: Platform | None = typer.Option(
        None,
        '-p',
        '--platform',
        help='If set, schedule this command on the specified platform instead of running inline.',
        show_default=True,
    ),
    debug: bool = typer.Option(
        False,
        '-d',
        '--debug',
        help='Enable Debugging Mode for the batch manager',
        show_default=True,
    ),
    vm_type: str | None = typer.Option(
        None, '--vm-type', help='Coiled VM type override (Coiled only).'
    ),
):
    """
    Aggregate geoparquet regions, reproject and write.
    """

    # Schedule if requested and not already inside a batch task
    if platform is not None and not _in_batch():
        config = load_config(env_file)
        manager = _get_manager(platform, debug)
        command = 'ocr aggregate'
        name = f'aggregate-geoparquet-{config.branch.value}'

        if platform == Platform.COILED:
            kwargs = {**_coiled_kwargs(config, env_file)}
            if vm_type:
                kwargs['vm_type'] = vm_type
        else:
            kwargs = {**_local_kwargs()}

        manager.submit_job(command=command, name=name, kwargs=kwargs)
        manager.wait_for_completion(exit_on_failure=True)
        return

    from ocr.pipeline.aggregate import aggregated_gpq

    config = load_config(env_file)
    aggregated_gpq(
        input_path=config.vector.region_geoparquet_uri,
        output_path=config.vector.building_geoparquet_uri,
    )


@app.command()
def aggregate_region_risk_summary_stats(
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
    platform: Platform | None = typer.Option(
        None,
        '-p',
        '--platform',
        help='If set, schedule this command on the specified platform instead of running inline.',
        show_default=True,
    ),
    debug: bool = typer.Option(
        False,
        '-d',
        '--debug',
        help='Enable Debugging Mode for the batch manager',
        show_default=True,
    ),
    vm_type: str | None = typer.Option(
        None, '--vm-type', help='Coiled VM type override (Coiled only).'
    ),
):
    """
    Generate statistical summaries at county and tract levels.
    """

    # Schedule if requested and not already inside a batch task
    if platform is not None and not _in_batch():
        config = load_config(env_file)
        manager = _get_manager(platform, debug)
        command = 'ocr aggregate-region-risk-summary-stats'
        name = f'create-aggregated-region-summary-stats-{config.branch.value}'

        if platform == Platform.COILED:
            kwargs = {**_coiled_kwargs(config, env_file)}
            if vm_type:
                kwargs['vm_type'] = vm_type
        else:
            kwargs = {**_local_kwargs()}

        manager.submit_job(command=command, name=name, kwargs=kwargs)
        manager.wait_for_completion(exit_on_failure=True)
        return

    from ocr.pipeline.fire_wind_risk_regional_aggregator import (
        compute_regional_fire_wind_risk_statistics,
    )

    config = load_config(env_file)

    compute_regional_fire_wind_risk_statistics(
        tracts_summary_stats_path=config.vector.tracts_summary_stats_uri,
        counties_summary_stats_path=config.vector.counties_summary_stats_uri,
        consolidated_buildings_path=config.vector.building_geoparquet_uri,
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
    platform: Platform | None = typer.Option(
        None,
        '-p',
        '--platform',
        help='If set, schedule this command on the specified platform instead of running inline.',
        show_default=True,
    ),
    debug: bool = typer.Option(
        False,
        '-d',
        '--debug',
        help='Enable Debugging Mode for the batch manager',
        show_default=True,
    ),
    vm_type: str | None = typer.Option(
        None, '--vm-type', help='Coiled VM type override (Coiled only).'
    ),
):
    """
    Create PMTiles for regional risk statistics (counties and tracts).
    """

    # Schedule if requested and not already inside a batch task
    if platform is not None and not _in_batch():
        config = load_config(env_file)
        manager = _get_manager(platform, debug)
        command = 'ocr create-regional-pmtiles'
        name = f'create-aggregated-region-pmtiles-{config.branch.value}'

        if platform == Platform.COILED:
            kwargs = {**_coiled_kwargs(config, env_file)}
            if vm_type:
                kwargs['vm_type'] = vm_type
        else:
            kwargs = {**_local_kwargs()}

        manager.submit_job(command=command, name=name, kwargs=kwargs)
        manager.wait_for_completion(exit_on_failure=True)
        return

    from ocr.pipeline.create_regional_pmtiles import create_regional_pmtiles

    config = load_config(env_file)

    create_regional_pmtiles(
        tracts_summary_stats_path=config.vector.tracts_summary_stats_uri,
        counties_summary_stats_path=config.vector.counties_summary_stats_uri,
        tract_pmtiles_output=config.vector.tracts_pmtiles_uri,
        county_pmtiles_output=config.vector.counties_pmtiles_uri,
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
    platform: Platform | None = typer.Option(
        None,
        '-p',
        '--platform',
        help='If set, schedule this command on the specified platform instead of running inline.',
        show_default=True,
    ),
    debug: bool = typer.Option(
        False,
        '-d',
        '--debug',
        help='Enable Debugging Mode for the batch manager',
        show_default=True,
    ),
    vm_type: str | None = typer.Option(
        None, '--vm-type', help='Coiled VM type override (Coiled only).'
    ),
):
    """
    Create PMTiles from the consolidated geoparquet file.
    """

    # Schedule if requested and not already inside a batch task
    if platform is not None and not _in_batch():
        config = load_config(env_file)
        manager = _get_manager(platform, debug)
        command = 'ocr create-pmtiles'
        name = f'create-pmtiles-{config.branch.value}'

        if platform == Platform.COILED:
            kwargs = {**_coiled_kwargs(config, env_file)}
            if vm_type:
                kwargs['vm_type'] = vm_type
        else:
            kwargs = {**_local_kwargs()}

        manager.submit_job(command=command, name=name, kwargs=kwargs)
        manager.wait_for_completion(exit_on_failure=True)
        return

    from ocr.pipeline.create_pmtiles import create_pmtiles

    config = load_config(env_file)

    create_pmtiles(
        input_path=config.vector.building_geoparquet_uri,
        output_path=config.vector.buildings_pmtiles_uri,
    )


ocr = typer.main.get_command(
    app
)  # this is needed to make the app available in the docs via mkdocs-click


if __name__ == '__main__':
    typer.run(run)
