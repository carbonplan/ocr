import os
import tempfile
import time
import typing
from pathlib import Path

import dotenv
import typer

from ocr.config import OCRConfig, load_config
from ocr.console import console
from ocr.deploy.managers import _get_manager
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


def _in_batch() -> bool:
    return os.environ.get('BATCH_ENV_FLAG') == '1'


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
    write_regional_stats: bool = typer.Option(
        False,
        '--write-regional-stats',
        help='Write aggregated statistical summaries for each region (one file per region type with stats like averages, medians, percentiles, and histograms)',
        show_default=True,
    ),
    write_regional_subsets: bool = typer.Option(
        False,
        '--write-regional-subsets',
        help='Write individual building-level data split by region (one file per GEOID containing raw building records)',
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
    process_retries: int = typer.Option(
        2,
        '--process-retries',
        min=0,
        help='Number of times to retry failed process-region tasks (Coiled only). 0 disables retries.',
        show_default=True,
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
        if write_regional_stats:
            parts += ['--write-regional-stats']
        if write_regional_subsets:
            parts += ['--write-regional-subsets']
        if wipe:
            parts += ['--wipe']

        # forward --env-file
        parts += ['--env-file', str(env_file)] if env_file else []

        command = ' '.join(parts)

        config = load_config(env_file)
        manager = _get_manager(dispatch_platform, config.debug)
        name = f'run-{config.environment.value}'

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

    # ------------- CONFIG ---------------

    config = load_config(env_file)
    # Defensive: sub-configs are populated in OCRConfig.model_post_init
    assert config.icechunk is not None and config.chunking is not None and config.vector is not None
    config.icechunk.init_repo()  # Ensure the Icechunk repo is initialized
    if wipe:
        config.icechunk.wipe()
        config.vector.wipe()

    if platform == Platform.COILED:
        # ------------- 01 AU ---------------

        # --- 01 Process Regions (with optional retries) ---
        COILED_SOFTWARE = os.environ.get('COILED_SOFTWARE_ENV_NAME')
        if COILED_SOFTWARE is None or not COILED_SOFTWARE.strip():
            console.log(
                '[red]Error: COILED_SOFTWARE_ENV_NAME environment variable is not set. '
                'This must be set to the name of a Coiled software environment with OCR installed. Proceeding with package sync...[/red]'
            )

        attempt = 0
        while True:
            attempt += 1
            # Use central config helper to resolve / validate region IDs
            region_status = config.select_region_ids(region_id, all_region_ids=all_region_ids)
            remaining_to_process = sorted(list(region_status.unprocessed_valid_region_ids))

            manager = _get_manager(Platform.COILED, config.debug)

            kwargs = _coiled_kwargs(config, env_file)
            # remove ntasks so we use map semantics
            kwargs.pop('ntasks', None)
            manager.submit_job(
                command=(
                    f'ocr process-region $COILED_BATCH_TASK_INPUT --risk-type {risk_type.value}'
                ),
                name=f'process-region-{config.environment.value}-attempt-{attempt}',
                kwargs={
                    **kwargs,
                    'map_over_values': remaining_to_process,
                    'software': COILED_SOFTWARE,
                },
            )
            completed, failed = manager.wait_for_completion(exit_on_failure=False)

            if not failed:
                break
            # map_over_values failure detection: we need to infer failures by difference
            # coiled batch currently only tracks job level; re-submit failed values if any remain
            # For now we conservatively retry all remaining values if any task failed.
            console.log(
                f'[yellow]Attempt {attempt} finished with failures. Retrying up to {process_retries} times.[/yellow]'
            )
            if attempt > process_retries:
                raise RuntimeError(
                    f'process-region mapping failed after {attempt} attempts. Failed job ids: {failed}'
                )
            # Retry all values (could refine by inspecting logs later)
            # small backoff
            time.sleep(5 * attempt)

        # ----------- 02 Aggregate -------------
        manager = _get_manager(Platform.COILED, config.debug)
        manager.submit_job(
            command='ocr aggregate',
            name=f'aggregate-geoparquet-{config.environment.value}',
            kwargs={
                **_coiled_kwargs(config, env_file),
                'vm_type': 'c8g.8xlarge',
                'scheduler_vm_type': 'c8g.8xlarge',
                'software': COILED_SOFTWARE,
            },
        )
        manager.wait_for_completion(exit_on_failure=True)

        if write_regional_stats:
            manager = _get_manager(Platform.COILED, config.debug)

            manager.submit_job(
                command='ocr write-aggregated-region-analysis-files',
                name=f'write-aggregated-region-analysis-files-{config.environment.value}',
                kwargs={
                    **_coiled_kwargs(config, env_file),
                    'vm_type': 'm8g.2xlarge',
                    'software': COILED_SOFTWARE,
                },
            )
        if write_regional_subsets:
            batch_manager = _get_manager(Platform.COILED, config.debug)

            batch_manager.submit_job(
                command='ocr write-per-region-analysis-files',
                name=f'write-per-region-analysis-files-{config.environment.value}',
                kwargs={
                    **_coiled_kwargs(config, env_file),
                    'vm_type': 'm8g.2xlarge',
                    'software': COILED_SOFTWARE,
                },
            )

        manager = _get_manager(Platform.COILED, config.debug)
        manager.submit_job(
            command='ocr aggregate-region-risk-summary-stats',
            name=f'create-aggregated-region-summary-stats-{config.environment.value}',
            kwargs={
                **_coiled_kwargs(config, env_file),
                'vm_type': 'c8g.8xlarge',
                'scheduler_vm_type': 'c8g.8xlarge',
                'software': COILED_SOFTWARE,
            },
        )
        manager.wait_for_completion(exit_on_failure=True)

        # create summary stats PMTiles layer
        manager = _get_manager(Platform.COILED, config.debug)
        manager.submit_job(
            command='ocr create-regional-pmtiles',
            name=f'create-aggregated-region-pmtiles-{config.environment.value}',
            kwargs={
                **_coiled_kwargs(config, env_file),
                'vm_type': 'c8g.8xlarge',
                'scheduler_vm_type': 'c8g.8xlarge',
                'disk_size': 250,
                'software': COILED_SOFTWARE,
            },
        )

        # ------------- 03  Tiles ---------------

        manager = _get_manager(Platform.COILED, config.debug)
        manager.submit_job(
            command='ocr create-pmtiles',
            name=f'create-pmtiles-{config.environment.value}',
            kwargs={
                **_coiled_kwargs(config, env_file),
                'vm_type': 'c8g.8xlarge',
                'scheduler_vm_type': 'c8g.8xlarge',
                'disk_size': 250,
                'software': COILED_SOFTWARE,
            },  # PMTiles creation needs more disk space
        )

        manager.wait_for_completion(exit_on_failure=True)

    elif platform == Platform.LOCAL:
        manager = _get_manager(Platform.LOCAL, config.debug)

        # Use central config helper to resolve / validate region IDs
        region_status = config.select_region_ids(region_id, all_region_ids=all_region_ids)
        remaining_to_process = sorted(list(region_status.unprocessed_valid_region_ids))

        for rid in remaining_to_process:
            manager.submit_job(
                command=f'ocr process-region {rid} --risk-type {risk_type.value}',
                name=f'process-region-{rid}-{config.environment.value}',
                kwargs={
                    **_local_kwargs(),
                },
            )
        manager.wait_for_completion(exit_on_failure=True)

        # Aggregate geoparquet regions
        manager = _get_manager(Platform.LOCAL, config.debug)
        manager.submit_job(
            command='ocr aggregate',
            name=f'aggregate-geoparquet-{config.environment.value}',
            kwargs={
                **_local_kwargs(),
            },
        )
        manager.wait_for_completion(exit_on_failure=True)

        if write_regional_stats:
            manager = _get_manager(Platform.LOCAL, config.debug)

            manager.submit_job(
                command='ocr write-aggregated-region-analysis-files',
                name=f'write-aggregated-region-analysis-files-{config.environment.value}',
                kwargs={
                    **_local_kwargs(),
                },
            )

        if write_regional_subsets:
            manager = _get_manager(Platform.LOCAL, config.debug)

            manager.submit_job(
                command='ocr write-per-region-analysis-files',
                name=f'write-per-region-analysis-files-{config.environment.value}',
                kwargs={
                    **_local_kwargs(),
                },
            )

        manager = _get_manager(Platform.LOCAL, config.debug)
        # Aggregate regional fire and wind risk statistics
        manager.submit_job(
            command='ocr aggregate-region-risk-summary-stats',
            name=f'create-aggregated-region-summary-stats-{config.environment.value}',
            kwargs={
                **_local_kwargs(),
            },
        )
        manager.wait_for_completion(exit_on_failure=True)

        # Create summary stats PMTiles layer
        manager = _get_manager(Platform.LOCAL, config.debug)
        manager.submit_job(
            command='ocr create-regional-pmtiles',
            name=f'create-aggregated-region-pmtiles-{config.environment.value}',
            kwargs={
                **_local_kwargs(),
            },
        )
        manager.wait_for_completion(exit_on_failure=True)

        # Create PMTiles from the consolidated geoparquet file
        manager = _get_manager(Platform.LOCAL, config.debug)
        manager.submit_job(
            command='ocr create-pmtiles',
            name=f'create-pmtiles-{config.environment.value}',
            kwargs={
                **_local_kwargs(),
            },
        )
        manager.wait_for_completion(exit_on_failure=True)

    if config.debug:
        # Print out the pretty paths
        console.log('Run complete. Current configuration paths:')
        config.pretty_paths()


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
    vm_type: str | None = typer.Option(
        None, '--vm-type', help='Coiled VM type override (Coiled only).'
    ),
    init_repo: bool = typer.Option(
        False, '--init-repo', help='Initialize Icechunk repository (if not already initialized).'
    ),
):
    """
    Calculate and write risk for a given region to Icechunk CONUS template.
    """

    # Schedule if requested and not already inside a batch task
    if platform is not None and not _in_batch():
        config = load_config(env_file)
        manager = _get_manager(platform, config.debug)
        command = f'ocr process-region {region_id} --risk-type {risk_type.value}'
        if init_repo:
            command += ' --init-repo'
        name = f'process-region-{region_id}-{config.environment.value}'

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
    if init_repo:
        config.icechunk.init_repo()

    calculate_risk(
        config=config,
        region_id=region_id,
        risk_type=risk_type,
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
        manager = _get_manager(platform, config.debug)
        command = 'ocr aggregate'
        name = f'aggregate-geoparquet-{config.environment.value}'

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
    aggregated_gpq(config=config)


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
    vm_type: str | None = typer.Option(
        None, '--vm-type', help='Coiled VM type override (Coiled only).'
    ),
):
    """
    Generate time-horizon based statistical summaries for county and tract level PMTiles creation
    """

    # Schedule if requested and not already inside a batch task
    if platform is not None and not _in_batch():
        config = load_config(env_file)
        manager = _get_manager(platform, config.debug)
        command = 'ocr aggregate-region-risk-summary-stats'
        name = f'create-aggregated-region-summary-stats-{config.environment.value}'

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

    compute_regional_fire_wind_risk_statistics(config=config)


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
        manager = _get_manager(platform, config.debug)
        command = 'ocr create-regional-pmtiles'
        name = f'create-aggregated-region-pmtiles-{config.environment.value}'

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

    create_regional_pmtiles(config=config)


@app.command()
def write_aggregated_region_analysis_files(
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
    vm_type: str | None = typer.Option(
        None, '--vm-type', help='Coiled VM type override (Coiled only).'
    ),
):
    """
    Write aggregated statistical summaries for each region (county and tract).

    Creates one file per region type containing aggregated statistics for ALL regions,
    including building counts, average/median risk values, percentiles (p90, p95, p99),
    and histograms. Outputs in geoparquet, geojson, and csv formats.
    """

    # Schedule if requested and not already inside a batch task
    if platform is not None and not _in_batch():
        config = load_config(env_file)
        manager = _get_manager(platform, config.debug)
        command = 'ocr write-aggregated-region-analysis-files'
        name = f'write-aggregated-region-analysis-files-{config.environment.value}'

        if platform == Platform.COILED:
            kwargs = {**_coiled_kwargs(config, env_file)}
            if vm_type:
                kwargs['vm_type'] = vm_type
        else:
            kwargs = {**_local_kwargs()}

        manager.submit_job(command=command, name=name, kwargs=kwargs)
        manager.wait_for_completion(exit_on_failure=True)
        return

    from ocr.pipeline.write_aggregated_region_analysis_files import (
        write_aggregated_region_analysis_files,
    )

    config = load_config(env_file)

    write_aggregated_region_analysis_files(config=config)


@app.command()
def write_per_region_analysis_files(
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
    vm_type: str | None = typer.Option(
        None, '--vm-type', help='Coiled VM type override (Coiled only).'
    ),
):
    """
    Write individual building-level fire risk data split by region (county and tract).

    Creates one file per GEOID containing raw building records with fire risk values,
    wind risk, and coordinates/geometry. Outputs thousands of files (one per county
    and one per tract) in csv and geojson formats for downloadable data subsets.
    """

    # Schedule if requested and not already inside a batch task
    if platform is not None and not _in_batch():
        config = load_config(env_file)
        manager = _get_manager(platform, config.debug)
        command = 'ocr write-per-region-analysis-files'
        name = f'write-per-region-analysis-files-{config.environment.value}'

        if platform == Platform.COILED:
            kwargs = {**_coiled_kwargs(config, env_file)}
            if vm_type:
                kwargs['vm_type'] = vm_type
        else:
            kwargs = {**_local_kwargs()}

        manager.submit_job(command=command, name=name, kwargs=kwargs)
        manager.wait_for_completion(exit_on_failure=True)
        return

    from ocr.pipeline.write_per_region_analysis_files import write_per_region_analysis_files

    config = load_config(env_file)

    write_per_region_analysis_files(config=config)


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
        manager = _get_manager(platform, config.debug)
        command = 'ocr create-pmtiles'
        name = f'create-pmtiles-{config.environment.value}'

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

    create_pmtiles(config=config)


ocr = typer.main.get_command(
    app
)  # this is needed to make the app available in the docs via mkdocs-click


if __name__ == '__main__':
    typer.run(run)
