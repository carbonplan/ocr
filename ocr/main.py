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
    debug: bool = False,
):
    from ocr.config import BatchJobs
    from ocr.template import TemplateConfig

    template_config = TemplateConfig()
    # TODO: Add options for already initalized repo
    template_config.init_icechunk_repo()
    template_config.create_template()

    batch_jobs = BatchJobs(region_id, run_on_coiled=run_on_coiled)
    batch_commands = batch_jobs.generate_batch_commands()
    print(batch_commands)
    for submit_command in batch_commands:
        print(f'submitting to coiled batch: {submit_command}')
        subprocess.Popen(submit_command, shell=True, cwd='.')


if __name__ == '__main__':
    main()
