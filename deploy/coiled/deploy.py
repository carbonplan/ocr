import click


@click.command(name='uv run deploy.py')
@click.option(
    '-r',
    '--region-id',
    multiple=True,  # this allows for multiple inputs: ex: uv run python deploy.py -r y2_x4 -r y2_x5
    help="region IDs. ex: 'y10_x2'",
)
@click.option('--branch', '-b', default='QA', help='data branch path [QA, prod]. Default is QA')
@click.option(
    '--wipe',
    '-w',
    is_flag=True,
    help='If True, wipes icechunk repo and vector data before initializing.',
)
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable Debugging, default False',
)
@click.option(
    '--summary-stats',
    '-s',
    is_flag=True,
    default=False,
    help='Adds in spatial summary aggregations.',
)
def main(
    region_id: tuple[str, ...],
    branch: str = 'QA',
    wipe: bool = False,
    debug: bool = False,
    summary_stats: bool = False,
):
    from ocr.batch import CoiledBatchManager
    from ocr.template import IcechunkConfig, VectorConfig

    # config_init applies any wipe and re-init opts
    IcechunkConfig(branch=branch, wipe=wipe).config_init()
    VectorConfig(branch=branch, wipe=wipe).config_init()

    # ------------- 01 AU ---------------

    shared_coiled_kwargs = {
        'ntasks': 1,
        'region': 'us-west-2',
        'forward_aws_credentials': True,
        'tag': {'Project': 'OCR'},
    }

    batch_manager_01 = CoiledBatchManager(debug=debug)

    # region_id is tuple
    for rid in region_id:
        batch_manager_01.submit_job(
            command=f'python ../../ocr/pipeline/01_Write_Region.py -r {rid} -b {branch}',
            name=f'process-region-{rid}-{branch}',
            kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.large'},
        )

    # this is a monitoring / blocking func. We should be able to block with this, then run 02, 03 etc.
    batch_manager_01.wait_for_completion()

    #     # ----------- 02 Pyramid -------------
    #     # NOTE: We need to do more work on pyramiding - currently excluded
    #     # This is non-blocking, since there are no post-pyramid dependent operations, so no wait_for_completion (I think)
    #     # if pyramid:
    #     #     batch_manager_pyarmid_02 = CoiledBatchManager(debug=debug)
    #     #     batch_manager_pyarmid_02.submit_job(
    #     #         command=f'python ../../ocr/pipeline/02_Pyramid.py -b {branch}',
    #     #         name=f'create-pyramid-{branch}',
    #     #     )
    #     # ----------- 02 Aggregate -------------
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
            command=f'python ../../ocr/pipeline/02_county_summary_stats.py -b {branch}',
            name=f'create-county-summary-stats-{branch}',
            kwargs={**shared_coiled_kwargs, 'vm_type': 'm8g.8xlarge'},
        )
        batch_manager_county_aggregation_01.wait_for_completion()
        # create county summary stats PMTiles layer
        batch_manager_county_tiles_02 = CoiledBatchManager(debug=debug)
        batch_manager_county_tiles_02.submit_job(
            command=f'../../ocr/pipeline/03_county_pmtiles.sh {branch}',
            name=f'create-county-pmtiles-{branch}',
            kwargs={
                **shared_coiled_kwargs,
                'vm_type': 'c7a.xlarge',
                'container': 'quay.io/carbonplan/ocr:latest',
            },
        )

    # # ------------- 03  Tiles ---------------
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


if __name__ == '__main__':
    main()
