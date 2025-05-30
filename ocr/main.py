import subprocess

import click


@click.command()
@click.option(
    '-r',
    '--region-id',
    multiple=True,  # this allows for multiple inputs: ex: uv run python ocr/main.py -r y10_x0 -r y0_x10
    help="region IDs. ex: 'y10_x2'",
)
@click.option('--all-regions', is_flag=True, help='run all region_ids')
@click.option('-c', '--run-on-coiled', is_flag=True, help='If True, run via coiled batch')
@click.option(
    '--batch-size',
    default=10,
    help='# of atomic units of work to submit to coiled batch. Hard max is 1000',
)
def main(
    region_id: tuple[str, ...],
    all_regions: bool = False,
    run_on_coiled: bool = False,
    batch_size: int = 10,
):
    # TODO: Add option for --all-regions. This can be sent to BatchJobs
    """Steps:
    1. init template - config.TemplateConfig().init_icechunk_repo()
    2. Check template ancestry - TODO: config.TemplateConfig().check_icechunk_ancestry()
    3. diff of regions to already written regions - TODO:
    """
    from ocr.config import BatchJobs

    coiled_batch_cmds = BatchJobs(region_id, run_on_coiled=run_on_coiled)
    batch_commands = coiled_batch_cmds.generate_batch_commands()
    print(batch_commands)
    for submit_command in batch_commands:
        print(submit_command)
        # print(f'submitting to coiled batch: {submit_command}')
        # subprocess.Popen(submit_command, shell=True, cwd='.')


if __name__ == '__main__':
    main()




# NEXT
# - try icechunk repo creation func
# - add click to 02_ (b/c 02 should still be an AU calc)
# - test running pipeline again
# - refactor ChunkingConfig to generalize to any dataset, not tied to USFS dataset
# - Start adding in Wind. 1. seperate out start. 2. ...
