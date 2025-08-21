import time
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from ocr.deploy.managers import AbstractBatchManager, LocalBatchManager


class TestAbstractBatchManager:
    """Test the abstract base class."""

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        manager = AbstractBatchManager()

        with pytest.raises(NotImplementedError):
            manager.submit_job('test command', 'test_name', {})

        with pytest.raises(NotImplementedError):
            manager.wait_for_completion()


class TestLocalBatchManager:
    """Test the LocalBatchManager class."""

    @pytest.fixture
    def manager(self):
        """Create a LocalBatchManager instance for testing."""
        return LocalBatchManager(max_workers=2, debug=True)

    def test_init_default_values(self):
        """Test manager initialization with default values."""
        manager = LocalBatchManager(debug=True)
        assert manager.status_check_int == 1
        assert manager.max_workers == 4
        assert manager.debug is True
        assert manager.jobs == {}
        assert manager._executor is not None
        assert isinstance(manager._executor, ThreadPoolExecutor)

    def test_init_custom_values(self):
        """Test manager initialization with custom values."""
        manager = LocalBatchManager(status_check_int=5, max_workers=8, debug=True)
        assert manager.status_check_int == 5
        assert manager.max_workers == 8
        assert manager.debug is True

    def test_submit_job_basic(self, manager):
        """Test basic job submission."""
        with patch.object(manager._executor, 'submit') as mock_submit:
            mock_future = Mock(spec=Future)
            mock_submit.return_value = mock_future

            job_id = manager.submit_job("echo 'hello'", 'test_job', {})

            assert job_id in manager.jobs
            job_info = manager.jobs[job_id]
            assert job_info['name'] == 'test_job'
            assert job_info['command'] == "echo 'hello'"
            assert job_info['state'] == 'queued'
            assert job_info['future'] == mock_future
            assert 'start_time' in job_info
            assert job_info['end_time'] is None
            assert job_info['return_code'] is None

    def test_submit_job_with_kwargs(self, manager):
        """Test job submission with subprocess kwargs."""
        with patch.object(manager._executor, 'submit') as mock_submit:
            mock_future = Mock(spec=Future)
            mock_submit.return_value = mock_future

            kwargs = {
                'env': {'PATH': '/usr/bin'},
                'cwd': '/tmp',
                'coiled_specific': 'ignored',  # Should be filtered out
            }

            manager.submit_job('ls', 'list_job', kwargs)

            # Check that submit was called correctly
            args, call_kwargs = mock_submit.call_args
            # ThreadPoolExecutor.submit is called with (fn, *args)
            # So args = (self._run_command, command, subprocess_kwargs)
            assert len(args) == 3
            method, command, subprocess_kwargs = args
            assert command == 'ls'
            assert subprocess_kwargs == {'env': {'PATH': '/usr/bin'}, 'cwd': '/tmp'}
            # Verify coiled-specific kwargs were filtered out
            assert 'coiled_specific' not in subprocess_kwargs

    def test_run_command_success(self, manager):
        """Test successful command execution."""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = 'success output'
            mock_result.stderr = ''
            mock_run.return_value = mock_result

            result = manager._run_command("echo 'test'", {})

            assert result['return_code'] == 0
            assert result['stdout'] == 'success output'
            assert result['stderr'] == ''
            mock_run.assert_called_once_with(
                "echo 'test'", shell=True, capture_output=True, text=True
            )

    def test_run_command_failure(self, manager):
        """Test failed command execution."""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stdout = ''
            mock_result.stderr = 'command failed'
            mock_run.return_value = mock_result

            result = manager._run_command('false', {})

            assert result['return_code'] == 1
            assert result['stdout'] == ''
            assert result['stderr'] == 'command failed'

    def test_run_command_exception(self, manager):
        """Test command execution with exception."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception('Process error')

            result = manager._run_command('invalid', {})

            assert result['return_code'] == -1
            assert result['stdout'] == ''
            assert result['stderr'] == 'Process error'

    def test_wait_for_completion_no_jobs(self, manager):
        """Test waiting for completion when no jobs exist."""
        with patch('ocr.deploy.managers.console') as mock_console:
            manager.wait_for_completion()
            mock_console.log.assert_called_with('No jobs to wait for')

    def test_wait_for_completion_successful_jobs(self, manager):
        """Test waiting for completion with successful jobs."""
        # Mock futures for completed jobs
        future1 = Mock(spec=Future)
        future1.done.return_value = True
        future1.result.return_value = {'return_code': 0, 'stdout': 'output1', 'stderr': ''}

        future2 = Mock(spec=Future)
        future2.done.return_value = True
        future2.result.return_value = {'return_code': 0, 'stdout': 'output2', 'stderr': ''}

        # Add jobs to manager
        manager.jobs = {
            'job1': {
                'name': 'test1',
                'command': 'echo test1',
                'state': 'queued',
                'future': future1,
                'start_time': time.time(),
                'end_time': None,
                'return_code': None,
                'stdout': None,
                'stderr': None,
            },
            'job2': {
                'name': 'test2',
                'command': 'echo test2',
                'state': 'queued',
                'future': future2,
                'start_time': time.time(),
                'end_time': None,
                'return_code': None,
                'stdout': None,
                'stderr': None,
            },
        }

        with patch('ocr.deploy.managers.console'):
            manager.wait_for_completion()

        # Check that jobs were updated
        assert manager.jobs['job1']['state'] == 'done'
        assert manager.jobs['job2']['state'] == 'done'
        assert manager.jobs['job1']['return_code'] == 0
        assert manager.jobs['job2']['return_code'] == 0

    def test_wait_for_completion_failed_jobs(self, manager):
        """Test waiting for completion with failed jobs."""
        future = Mock(spec=Future)
        future.done.return_value = True
        future.result.return_value = {'return_code': 1, 'stdout': '', 'stderr': 'error'}

        manager.jobs = {
            'job1': {
                'name': 'test1',
                'command': 'false',
                'state': 'queued',
                'future': future,
                'start_time': time.time(),
                'end_time': None,
                'return_code': None,
                'stdout': None,
                'stderr': None,
            }
        }

        with patch('ocr.deploy.managers.console'):
            manager.wait_for_completion()

        assert manager.jobs['job1']['state'] == 'failed'
        assert manager.jobs['job1']['return_code'] == 1

    def test_wait_for_completion_exit_on_failure(self, manager):
        """Test exit_on_failure behavior."""
        future = Mock(spec=Future)
        future.done.return_value = True
        future.result.return_value = {'return_code': 1, 'stdout': '', 'stderr': 'error'}

        manager.jobs = {
            'job1': {
                'name': 'test1',
                'command': 'false',
                'state': 'queued',
                'future': future,
                'start_time': time.time(),
                'end_time': None,
                'return_code': None,
                'stdout': None,
                'stderr': None,
            }
        }

        with patch('ocr.deploy.managers.console'):
            with pytest.raises(Exception, match='Job job1 failed with return code 1'):
                manager.wait_for_completion(exit_on_failure=True)

    def test_wait_for_completion_exception_handling(self, manager):
        """Test exception handling during job completion."""
        future = Mock(spec=Future)
        future.done.return_value = True
        future.result.side_effect = Exception('Future exception')

        manager.jobs = {
            'job1': {
                'name': 'test1',
                'command': 'test',
                'state': 'queued',
                'future': future,
                'start_time': time.time(),
                'end_time': None,
                'return_code': None,
                'stdout': None,
                'stderr': None,
            }
        }

        with patch('ocr.deploy.managers.console'):
            manager.wait_for_completion()

        assert manager.jobs['job1']['state'] == 'failed'

    def test_wait_for_completion_running_jobs(self, manager):
        """Test handling of running jobs."""
        future = Mock(spec=Future)
        future.done.side_effect = [False, False, True]  # Running, then done
        future.running.side_effect = [True, True, False]
        future.result.return_value = {'return_code': 0, 'stdout': 'done', 'stderr': ''}

        manager.jobs = {
            'job1': {
                'name': 'test1',
                'command': 'sleep 1',
                'state': 'queued',
                'future': future,
                'start_time': time.time(),
                'end_time': None,
                'return_code': None,
                'stdout': None,
                'stderr': None,
            }
        }

        with patch('ocr.deploy.managers.console'), patch('time.sleep') as mock_sleep:
            manager.wait_for_completion()

        # Should have slept while waiting
        assert mock_sleep.call_count >= 1
        assert manager.jobs['job1']['state'] == 'done'

    def test_debug_mode_logging(self, manager):
        """Test debug mode logging for failed jobs."""
        manager.debug = True
        future = Mock(spec=Future)
        future.done.return_value = True
        future.result.return_value = {'return_code': 1, 'stdout': '', 'stderr': 'debug error'}

        manager.jobs = {
            'job1': {
                'name': 'test1',
                'command': 'false',
                'state': 'queued',
                'future': future,
                'start_time': time.time(),
                'end_time': None,
                'return_code': None,
                'stdout': None,
                'stderr': None,
            }
        }

        with patch('ocr.deploy.managers.console') as mock_console:
            manager.wait_for_completion()

        # Check that debug information was logged
        debug_calls = [
            call
            for call in mock_console.log.call_args_list
            if 'failed with return code' in str(call)
        ]
        assert len(debug_calls) > 0

    def test_executor_cleanup(self, manager):
        """Test that executor is properly cleaned up."""
        # Mock the executor's shutdown method
        with patch.object(manager._executor, 'shutdown') as mock_shutdown:
            # Simulate deletion
            manager.__del__()

            # Executor should be shut down
            mock_shutdown.assert_called_once_with(wait=False)

    def test_multiple_job_submission(self, manager):
        """Test submitting multiple jobs."""
        with patch.object(manager._executor, 'submit') as mock_submit:
            mock_submit.return_value = Mock(spec=Future)

            job_ids = []
            for i in range(3):
                job_id = manager.submit_job(f'echo {i}', f'job_{i}', {})
                job_ids.append(job_id)

            assert len(manager.jobs) == 3
            assert all(job_id in manager.jobs for job_id in job_ids)
            assert mock_submit.call_count == 3

    def test_unique_job_ids(self, manager):
        """Test that job IDs are unique."""
        with patch.object(manager._executor, 'submit') as mock_submit:
            mock_submit.return_value = Mock(spec=Future)

            job_ids = set()
            for i in range(10):
                job_id = manager.submit_job(f'echo {i}', f'job_{i}', {})
                job_ids.add(job_id)

            # All job IDs should be unique
            assert len(job_ids) == 10
            assert len(manager.jobs) == 10
