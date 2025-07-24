import time
from dataclasses import dataclass, field

import click
import coiled


@dataclass
class CoiledBatchManager:
    debug: bool = False
    status_check_int: int = 10
    job_limit: int = 1000  # set in https://docs.coiled.io/_modules/coiled/batch.html#status - any downside to a big #?
    job_ids: list = field(default_factory=list)

    def submit_job(self, command, name, kwargs):
        batch_result = coiled.batch.run(command=command, name=name, **kwargs)
        job_id = batch_result['job_id']
        self.job_ids.append(job_id)
        return job_id

    def wait_for_completion(self, exit_on_failure=False):
        """Wait for all tracked jobs to complete."""

        print(f'Waiting for {len(self.job_ids)} jobs to complete...')

        # sets instead of lists so we don't add duplicates on each check
        completed = set()
        failed = set()
        while len(completed) + len(failed) < len(self.job_ids):
            # TODO/ Q: are there other status's for coiled batch jobs. If so, this might stall.
            all_jobs = coiled.batch.list_jobs(limit=self.job_limit)
            for job in all_jobs:
                job_id = job.get('id')
                if job_id in self.job_ids and job_id not in completed and job_id not in failed:
                    # if job_id is one of submitted job_ids, incase someone else submits... and not in failed or completed.
                    state = job.get('state')
                    print(f'job id: {job_id} and state: {state}')
                    if state == 'done':
                        print(f'{job_id} success')
                        completed.add(job_id)
                    elif state in ('failed', 'error', 'done (errors)'):
                        print(f'{job_id} failed')
                        failed.add(job_id)
                        if exit_on_failure:
                            raise Exception(f'{job_id} failed, and exit_on_failure == True. ')

                    elif state == 'queued':
                        print(f'{job_id} is queued')
                    elif state == 'running':
                        print(f'{job_id} is running')
                    else:
                        raise NotImplementedError(
                            f'state: {state} for job_id: {job_id} is not done, failed, queued or running. We need to add more checks!'
                        )
            print(
                f'\n ---------status checked every {self.status_check_int} seconds ----------- \n'
            )

            if len(completed) + len(failed) < len(self.job_ids):
                print(f'{len(completed)} jobs have completed out of {len(self.job_ids)}')
                time.sleep(self.status_check_int)
        # TODO: switch to logging!
        print(
            f'\n ---------------- \n all jobs finished running \n completed: {len(completed)}, {completed} \n  failed: {len(failed)}, {failed} \n ------------'
        )


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
    from ocr.chunking_config import ChunkingConfig
    from ocr.template import IcechunkConfig, VectorConfig, get_commit_messages_ancestry

    # ------------- CONFIG ---------------
    config = ChunkingConfig()
    VectorConfig(branch=branch, wipe=wipe)
    icechunk_config = IcechunkConfig(branch=branch, wipe=wipe)

    icechunk_repo_and_session = icechunk_config.repo_and_session()
    valid_region_ids = set(region_id).intersection(config.valid_region_ids)
    region_ids_in_ancestry = get_commit_messages_ancestry(icechunk_repo_and_session['repo'])
    valid_region_ids = valid_region_ids.difference(region_ids_in_ancestry)

    if len(valid_region_ids) == 0:
        raise ValueError(
            f'There are no valid region_ids present. Supplied region_ids: {region_id} are already in the icechunk ancestry or are invalid region ids.'
        )

    shared_coiled_kwargs = {
        'ntasks': 1,
        'region': 'us-west-2',
        'forward_aws_credentials': True,
        'tag': {'Project': 'OCR'},
    }

    # ------------- 01 AU ---------------

    batch_manager_01 = CoiledBatchManager(debug=debug)

    # region_id is tuple
    for rid in valid_region_ids:
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


if __name__ == '__main__':
    main()
