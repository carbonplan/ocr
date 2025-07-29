import time
import typing

import coiled
import pydantic
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ocr.console import console


class AbstractBatchManager(pydantic.BaseModel):
    """
    Abstract base class for batch managers.
    """

    debug: bool = False

    def submit_job(self, command: str, name: str, kwargs: dict[str, typing.Any]):
        """
        Wait for the batch job to complete.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    def wait_for_completion(self, exit_on_failure: bool = False):
        """
        Get the batch ID.
        """
        raise NotImplementedError('Subclasses must implement this method.')


class CoiledBatchManager(AbstractBatchManager):
    """
    Coiled batch manager for managing batch jobs.
    """

    status_check_int: int = 10
    job_limit: int = 1000  # set in https://docs.coiled.io/_modules/coiled/batch.html#status - any downside to a big #?
    job_ids: list[str] = pydantic.Field(default_factory=list)

    def submit_job(self, command: str, name: str, kwargs: dict[str, typing.Any]) -> str:
        console.log(f'Submitting job: {name} with command: {command} and kwargs: {kwargs}')
        batch_result = coiled.batch.run(command=command, name=name, **kwargs)
        job_id = batch_result['job_id']
        self.job_ids.append(job_id)
        return job_id

    def wait_for_completion(self, exit_on_failure: bool = False):
        """Wait for all tracked jobs to complete."""

        # sets instead of lists so we don't add duplicates on each check
        completed: set = set()
        failed: set = set()

        with Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            main_task = progress.add_task(
                f'Processing {len(self.job_ids)} jobs', total=len(self.job_ids)
            )

            while len(completed) + len(failed) < len(self.job_ids):
                # TODO/ Q: are there other status's for coiled batch jobs. If so, this might stall.
                all_jobs = coiled.batch.list_jobs(limit=self.job_limit)
                current_states = {'queued': 0, 'running': 0, 'done': 0, 'failed': 0}
                for job in all_jobs:
                    job_id = job.get('id')
                    if job_id in self.job_ids and job_id not in completed and job_id not in failed:
                        # if job_id is one of submitted job_ids, incase someone else submits... and not in failed or completed.
                        state = job.get('state')

                        if state == 'done':
                            completed.add(job_id)
                            current_states['done'] += 1
                        elif state in ('failed', 'error', 'done (errors)'):
                            failed.add(job_id)
                            current_states['failed'] += 1
                            if exit_on_failure:
                                raise Exception(f'{job_id} failed, and exit_on_failure == True. ')

                        elif state == 'queued':
                            current_states['queued'] += 1
                        elif state == 'running':
                            current_states['running'] += 1
                        else:
                            progress.stop()
                            raise NotImplementedError(
                                f'state: {state} for job_id: {job_id} is not done, failed, queued or running. We need to add more checks!'
                            )

                total_processed = len(completed) + len(failed)
                progress.update(
                    main_task,
                    completed=total_processed,
                    description=f'Jobs - ✅ {len(completed)} done, ❌ {len(failed)} failed, 🏃 {current_states["running"]} running, ⏳ {current_states["queued"]} queued',
                )
                if len(completed) + len(failed) < len(self.job_ids):
                    time.sleep(self.status_check_int)

        table = Table(title='Job Completion Summary')
        table.add_column('Status', style='cyan')
        table.add_column('Count', justify='right', style='magenta')
        table.add_column('Job IDs', style='green')
        table.add_row(
            '✅ Completed',
            str(len(completed)),
            ', '.join(list(completed)[:5]) + ('...' if len(completed) > 5 else ''),
        )

        table.add_row(
            '❌ Failed',
            str(len(failed)),
            ', '.join(list(failed)[:5]) + ('...' if len(failed) > 5 else ''),
        )

        console.print('\n')
        console.print(table)
        console.print(f'\n[bold green]All {len(self.job_ids)} jobs finished![/bold green]')
