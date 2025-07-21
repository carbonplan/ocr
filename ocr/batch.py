import time
from dataclasses import dataclass, field

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
                    elif state in ('failed', 'error'):
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
