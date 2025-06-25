import click


@click.command(name='uv run main.py')
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable Debugging, default False',
)
@click.option(
    '-r',
    '--region-id',
    multiple=True,  # this allows for multiple inputs: ex: uv run python ocr/main.py -r y10_x0 -r y0_x10
    help="region IDs. ex: 'y10_x2'",
)
@click.option('--branch', '-b', default='QA', help='data branch path [QA, prod]. Default is QA')
@click.option(
    '--wipe',
    '-w',
    is_flag=True,
    help='If True, wipes icechunk repo and vector data before initializing.',
)
def main(
    region_id: tuple[str, ...],
    branch: str = 'QA',
    wipe: bool = False,
    debug: bool = False,
):
    # from ocr.config import BatchJobs
    from ocr.batch import CoiledBatchManager
    from ocr.template import IcechunkConfig, VectorConfig

    # config_init applies any wipe and re-init opts
    IcechunkConfig(branch=branch, wipe=wipe).config_init()
    VectorConfig(branch=branch, wipe=wipe).config_init()

    # ------------- 01 ---------------

    batch_manger_01 = CoiledBatchManager(debug=debug)
    # region_id is tuple
    for rid in region_id:
        batch_manger_01.submit_job(
            command=f'python pipeline/01_Write_Region.py -r {rid} -b {branch}',
            name=f'process-region-{rid}-{branch}',
        )

    # this is a monitoring / blocking func. We should be able to block with this, then run 02, 03 etc.
    batch_manger_01.wait_for_completion()

    # ------------- 02 ---------------
    batch_manger_02 = CoiledBatchManager(debug=debug)
    batch_manger_02.submit_job(
        command=f'python pipeline/02_Aggregate.py -b {branch}',
        name=f'aggregate-geoparuqet-{branch}',
    )
    batch_manger_02.wait_for_completion()

    # ------------- 03 ---------------
    batch_manger_03 = CoiledBatchManager(debug=debug)
    batch_manger_03.submit_job(
        command='pipeline/03_Tiles.sh',
        name=f'create-pmtiles-{branch}',
    )
    batch_manger_03.wait_for_completion()


if __name__ == '__main__':
    main()
