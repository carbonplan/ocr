import subprocess
import time
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor

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
from ocr.types import Platform


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
        if self.debug:
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
                                raise Exception(
                                    f'{job_id} failed because {job["n_tasks_failed"]} / {job["n_tasks"]} tasks failed and exit_on_failure == True.\nJob details: {job}'
                                )

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
        if self.debug:
            table = Table(title='Job Completion Summary')
            table.add_column('Status', style='cyan')
            table.add_column('Count', justify='right', style='magenta')
            table.add_column('Job IDs', style='green')
            table.add_row(
                '✅ Completed',
                str(len(completed)),
                ', '.join([str(job_id) for job_id in list(completed)[:5]])
                + ('...' if len(completed) > 5 else ''),
            )

            table.add_row(
                '❌ Failed',
                str(len(failed)),
                ', '.join([str(job_id) for job_id in list(failed)[:5]])
                + ('...' if len(failed) > 5 else ''),
            )

            console.print('\n')
            console.print(table)
            console.print(f'\n[bold green]All {len(self.job_ids)} jobs finished![/bold green]')


class LocalBatchManager(AbstractBatchManager):
    """
    Local batch manager for running jobs locally using subprocess.
    """

    status_check_int: int = 1  # Check more frequently for local jobs
    max_workers: int = 4  # Number of concurrent local processes
    jobs: dict[str, dict] = pydantic.Field(default_factory=dict)
    _executor: ThreadPoolExecutor | None = None

    def model_post_init(self, __context):
        """Initialize the thread pool executor after model creation."""
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def __del__(self):
        """Clean up the executor when the manager is destroyed."""
        if self._executor:
            self._executor.shutdown(wait=False)

    def submit_job(self, command: str, name: str, kwargs: dict[str, typing.Any]) -> str:
        """Submit a job to run locally."""
        job_id = str(uuid.uuid4())
        if self.debug:
            console.log(f'Submitting local job: {name} (ID: {job_id}) with command: {command}')

        # Extract relevant kwargs for subprocess (ignore coiled-specific ones)
        subprocess_kwargs = {}
        if 'env' in kwargs:
            subprocess_kwargs['env'] = kwargs['env']
        if 'cwd' in kwargs:
            subprocess_kwargs['cwd'] = kwargs['cwd']

        # Submit the job to the thread pool
        future = self._executor.submit(self._run_command, command, subprocess_kwargs)

        self.jobs[job_id] = {
            'name': name,
            'command': command,
            'state': 'queued',
            'future': future,
            'start_time': time.time(),
            'end_time': None,
            'return_code': None,
            'stdout': None,
            'stderr': None,
        }

        return job_id

    def _run_command(self, command: str, subprocess_kwargs: dict) -> dict:
        """Run a command using subprocess."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, **subprocess_kwargs
            )
            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
        except Exception as e:
            return {
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
            }

    def wait_for_completion(self, exit_on_failure: bool = False):
        """Wait for all tracked jobs to complete."""
        if not self.jobs:
            if self.debug:
                console.log('No jobs to wait for')
            return

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
                f'Processing {len(self.jobs)} local jobs', total=len(self.jobs)
            )

            while len(completed) + len(failed) < len(self.jobs):
                current_states = {'queued': 0, 'running': 0, 'done': 0, 'failed': 0}

                for job_id, job_info in self.jobs.items():
                    if job_id in completed or job_id in failed:
                        continue

                    future = job_info['future']

                    if future.done():
                        # Job has completed, get the result
                        try:
                            result = future.result()
                            job_info['end_time'] = time.time()
                            job_info['return_code'] = result['return_code']
                            job_info['stdout'] = result['stdout']
                            job_info['stderr'] = result['stderr']

                            if result['return_code'] == 0:
                                job_info['state'] = 'done'
                                completed.add(job_id)
                                current_states['done'] += 1
                            else:
                                job_info['state'] = 'failed'
                                failed.add(job_id)
                                current_states['failed'] += 1

                                if self.debug:
                                    console.log(
                                        f'Job {job_id} failed with return code {result["return_code"]}'
                                    )
                                    console.log(f'stderr: {result["stderr"]}')

                                if exit_on_failure:
                                    raise Exception(
                                        f'Job {job_id} failed with return code {result["return_code"]}: {result["stderr"]}'
                                    )

                        except Exception as e:
                            job_info['state'] = 'failed'
                            job_info['end_time'] = time.time()
                            failed.add(job_id)
                            current_states['failed'] += 1

                            if self.debug:
                                console.log(f'Job {job_id} failed with exception: {e}')

                            if exit_on_failure:
                                raise Exception(f'Job {job_id} failed with exception: {e}')

                    elif future.running():
                        job_info['state'] = 'running'
                        current_states['running'] += 1
                    else:
                        # Still queued
                        current_states['queued'] += 1

                total_processed = len(completed) + len(failed)
                progress.update(
                    main_task,
                    completed=total_processed,
                    description=f'Jobs - ✅ {len(completed)} done, ❌ {len(failed)} failed, 🏃 {current_states["running"]} running, ⏳ {current_states["queued"]} queued',
                )

                if len(completed) + len(failed) < len(self.jobs):
                    time.sleep(self.status_check_int)

        if self.debug:
            # Display summary table
            table = Table(title='Local Job Completion Summary')
            table.add_column('Status', style='cyan')
            table.add_column('Count', justify='right', style='magenta')
            table.add_column('Job IDs', style='green')

            completed_ids = [job_id for job_id in completed]
            failed_ids = [job_id for job_id in failed]

            table.add_row(
                '✅ Completed',
                str(len(completed)),
                ', '.join(completed_ids[:5]) + ('...' if len(completed) > 5 else ''),
            )

            table.add_row(
                '❌ Failed',
                str(len(failed)),
                ', '.join(failed_ids[:5]) + ('...' if len(failed) > 5 else ''),
            )

            console.print('\n')
            console.print(table)
            console.print(f'\n[bold green]All {len(self.jobs)} local jobs finished![/bold green]')

        # Cleanup executor
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


def _get_manager(platform: Platform, debug: bool):
    if platform == Platform.COILED:
        return CoiledBatchManager(debug=debug)
    elif platform == Platform.LOCAL:
        return LocalBatchManager(debug=debug)
    else:
        raise ValueError(f'Unknown platform: {platform}')
