import subprocess

import click


@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help='Enable Debugging, default False',
)
@click.command()
@click.option(
    '-r',
    '--region-id',
    multiple=True,  # this allows for multiple inputs: ex: uv run python ocr/main.py -r y10_x0 -r y0_x10
    help="region IDs. ex: 'y10_x2'",
)
@click.option('-c', '--run-on-coiled', is_flag=True, help='If True, run via coiled batch')
@click.option('--branch', '-b', default='QA', help='data branch path [QA, prod]. Default is QA')
@click.option(
    '--wipe',
    '-w',
    is_flag=True,
    help='If True, wipes icechunk repo and vector data before initializing.',
)
def main(
    region_id: tuple[str, ...],
    run_on_coiled: bool = False,
    branch: bool = False,
    wipe: bool = False,
    debug: bool = False,
):
    from ocr.config import BatchJobs
    from ocr.template import IcechunkConfig, VectorConfig

    # config_init applies any wipe and re-init opts
    IcechunkConfig(branch=branch, wipe=wipe).config_init()
    VectorConfig(branch=branch, wipe=wipe).config_init()

    batch_jobs = BatchJobs(region_id, run_on_coiled=run_on_coiled)
    batch_commands = batch_jobs.generate_batch_commands(branch=branch)
    print(batch_commands)
    for submit_command in batch_commands:
        print(f'submitting: {submit_command}')
        subprocess.Popen(submit_command, shell=True, cwd='.')


if __name__ == '__main__':
    main()
