import typer

from ocr.types import Branch, Platform

app = typer.Typer(help='Run OCR deployment pipeline on Coiled')


@app.command()
def main(
    region_id: list[str] | None = typer.Option(
        None, '-r', '--region-id', help='Region IDs to process, e.g., y10_x2'
    ),
    all_region_ids: bool = typer.Option(
        False,
        '--all-region-ids',
        help='Process all valid region IDs',
        show_default=True,
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
    from ocr.template import IcechunkConfig, VectorConfig, get_commit_messages_ancestry

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
        from managers import CoiledBatchManager

        shared_coiled_kwargs = {
            'ntasks': 1,
            'region': 'us-west-2',
            'forward_aws_credentials': True,
            'tag': {'Project': 'OCR'},
        }

        # ------------- 01 AU ---------------

        batch_manager_01 = CoiledBatchManager(debug=debug)

        # region_id is tuple
        for rid in unprocessed_valid_region_ids:
            batch_manager_01.submit_job(
                command=f'python ../../ocr/pipeline/01_Write_Region.py -r {rid} -b {branch}',
                name=f'process-region-{rid}-{branch}',
                kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.large'},
            )

        # # this is a monitoring / blocking func. We should be able to block with this, then run 02, 03 etc.
        batch_manager_01.wait_for_completion()

        # ----------- 02 Aggregate -------------
        batch_manager_aggregate_02 = CoiledBatchManager(debug=debug)
        batch_manager_aggregate_02.submit_job(
            command=f'python ../../ocr/pipeline/02_Aggregate.py -b {branch}',
            name=f'aggregate-geoparuqet-{branch}',
            kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.large'},
        )
        batch_manager_aggregate_02.wait_for_completion()

        if summary_stats:
            batch_manager_county_aggregation_01 = CoiledBatchManager(debug=debug)
            batch_manager_county_aggregation_01.submit_job(
                command=f'python ../../ocr/pipeline/02_aggregated_region_summary_stats.py -b {branch}',
                name=f'create-county-summary-stats-{branch}',
                kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.6xlarge'},
            )
            batch_manager_county_aggregation_01.wait_for_completion()

            # create summary stats PMTiles layer
            batch_manager_county_tiles_02 = CoiledBatchManager(debug=debug)
            batch_manager_county_tiles_02.submit_job(
                command=f'../../ocr/pipeline/03_aggregated_region_pmtiles.sh {branch}',
                name=f'create-county-pmtiles-{branch}',
                kwargs={
                    **shared_coiled_kwargs,
                    'vm_type': 'c7a.4xlarge',
                    'container': 'quay.io/carbonplan/ocr:latest',
                },
            )

        # ------------- 03  Tiles ---------------
        batch_manager_03 = CoiledBatchManager(debug=debug)
        batch_manager_03.submit_job(
            command=f'../../ocr/pipeline/03_Tiles.sh {branch}',
            name=f'create-pmtiles-{branch}',
            kwargs={
                **shared_coiled_kwargs,
                'vm_type': 'c7a.xlarge',
                'container': 'quay.io/carbonplan/ocr:latest',
            },
        )

        batch_manager_03.wait_for_completion()

    elif platform == Platform.LOCAL:
        raise NotImplementedError(
            'Local platform is not implemented yet. Please use Coiled for now.'
        )


if __name__ == '__main__':
    typer.run(main)
