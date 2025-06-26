import click


@click.command(name='uv run deploy.py')
@click.option(
    '-r',
    '--region-id',
    multiple=True,  # this allows for multiple inputs: ex: uv run python deploy.py -r y10_x0 -r y0_x10
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
    '-p',
    '--pyramid',
    is_flag=True,
    default=False,
    help='Generate ndpyramid, default False',
)
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable Debugging, default False',
)
def main(
    region_id: tuple[str, ...],
    branch: str = 'QA',
    pyramid: bool = False,
    wipe: bool = False,
    debug: bool = False,
):
    # from ocr.config import BatchJobs
    from ocr.batch import CoiledBatchManager
    from ocr.template import IcechunkConfig, VectorConfig

    # config_init applies any wipe and re-init opts
    IcechunkConfig(branch=branch, wipe=wipe).config_init()
    VectorConfig(branch=branch, wipe=wipe).config_init()

    # ------------- 01 AU ---------------

    batch_manager_01 = CoiledBatchManager(debug=debug)
    # region_id is tuple
    for rid in region_id:
        batch_manager_01.submit_job(
            command=f'python pipeline/01_Write_Region.py -r {rid} -b {branch}',
            name=f'process-region-{rid}-{branch}',
        )

    # this is a monitoring / blocking func. We should be able to block with this, then run 02, 03 etc.
    batch_manager_01.wait_for_completion()

    # ----------- 02 Pyramid -------------
    # This is non-blocking, since there are no post-pyramid dependent operations, so no wait_for_completion (I think)
    if pyramid:
        batch_manager_pyarmid_02 = CoiledBatchManager(debug=debug)
        batch_manager_pyarmid_02.submit_job(
            command=f'python pipeline/02_Pyramid.py -b {branch}',
            name=f'create-pyramid-{branch}',
        )
    # ----------- 02 Aggregate -------------
    batch_manager_aggregate_02 = CoiledBatchManager(debug=debug)
    batch_manager_aggregate_02.submit_job(
        command=f'python pipeline/02_Aggregate.py -b {branch}',
        name=f'aggregate-geoparuqet-{branch}',
    )
    batch_manager_aggregate_02.wait_for_completion()

    # ------------- 03  Tiles ---------------
    batch_manager_03 = CoiledBatchManager(debug=debug)
    batch_manager_03.submit_job(
        command=f'pipeline/03_Tiles.sh {branch}',
        name=f'create-pmtiles-{branch}',
    )
    # We probably don't need this, but we do get reporting with it, which is kind of nice.
    batch_manager_03.wait_for_completion()


if __name__ == '__main__':
    main()
