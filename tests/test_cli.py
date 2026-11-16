import dataclasses
import json
from pathlib import Path
from unittest.mock import MagicMock

import coiled
import pytest
from typer.testing import CliRunner

from ocr import cli, stages
from ocr.buildings import PointSamplingStrategy
from ocr.config import Risk
from ocr.regions import RegionLevel
from ocr.stages import RISKS, STAGES


@pytest.fixture(autouse=True)
def inputs_present(monkeypatch: pytest.MonkeyPatch) -> None:
    # Most tests run a single stage; skip the storage check (`test_preflight*` covers it).
    monkeypatch.setattr(cli, 'preflight', lambda *_: [])


def jobs(state: str, exit_code: int | None = None, cluster_state: str = 'ready') -> list[dict]:
    return [{'cluster_state': cluster_state, 'tasks': [{'state': state, 'exit_code': exit_code}]}]


DONE = jobs('done', 0)


@pytest.fixture
def batch(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    m = MagicMock()
    m.run.return_value = {'job_id': 1, 'cluster_id': 2}
    m.status.return_value = DONE
    monkeypatch.setattr(coiled, 'batch', m)
    monkeypatch.setattr(cli, 'POLL_SECONDS', 0)
    return m


def invoke(*args: str):
    return CliRunner().invoke(cli.app, ['stage', '--version', 'v2.0.0', '--coiled', 'flood', *args])


def test_coiled_job_config(batch: MagicMock) -> None:
    r = invoke('buildings', '--vm', 'r8g.8xlarge')
    assert r.exit_code == 0, r.output
    kw = batch.run.call_args.kwargs
    assert kw['command'] == [
        *('ocr', 'stage', '--version', 'v2.0.0', '--prefix', 'output', 'flood', 'buildings')
    ]
    assert kw['vm_type'] == ['r8g.8xlarge']
    assert kw['name'] == 'ocr-flood-buildings-v2-0-0'  # Coiled rejects '.'
    assert kw['disk_size'] == '50 GiB' and kw['arm'] is True


def test_coiled_failure_exits_nonzero(batch: MagicMock) -> None:
    batch.status.side_effect = [jobs('running'), jobs('done', 1)]
    r = invoke('buildings')
    assert r.exit_code != 0


@pytest.mark.parametrize(
    ('status', 'state'),
    [
        ([], None),
        (jobs('pending'), None),
        (jobs('done', 0), 'done'),
        (jobs('done', 1), 'error'),
        (jobs('pending', cluster_state='error'), 'error'),
        (jobs('running', cluster_state='stopped'), 'error'),
        (jobs('done', 0, cluster_state='stopped'), 'done'),
        (jobs('error'), 'error'),
    ],
)
def test_job_state(status: list[dict], state: str | None) -> None:
    assert cli.job_state(status) == state


def two(a: tuple[str, int | None], b: tuple[str, int | None], cluster: str = 'ready') -> list[dict]:
    tasks = [{'state': s, 'exit_code': e} for s, e in (a, b)]
    return [{'cluster_state': cluster, 'tasks': tasks}]


@pytest.mark.parametrize(
    ('status', 'finished'),
    [
        ([], False),
        (jobs('pending'), False),
        # A failed task doesn't end a map job while others still run (unlike job_state).
        (two(('error', 1), ('running', None)), False),
        (two(('done', 1), ('running', None)), False),
        (two(('error', 1), ('done', 0)), True),
        (two(('done', 0), ('done', 0)), True),
        # Cluster gone (job_timeout, lost VM): finished, the next round re-plans.
        (two(('done', 0), ('running', None), cluster='stopped'), True),
        (jobs('pending', cluster_state='error'), True),
    ],
)
def test_job_finished(status: list[dict], finished: bool) -> None:
    assert cli.job_finished(status) is finished


def test_ctrl_c_deletes_cluster(batch: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    batch.status.side_effect = [jobs('running'), KeyboardInterrupt]
    deleted = MagicMock()
    monkeypatch.setattr(coiled, 'delete_cluster', deleted)
    assert invoke('buildings').exit_code != 0
    deleted.assert_called_once_with(cluster_id=2)


def test_failure_stops_later_stages(batch: MagicMock) -> None:
    batch.status.return_value = jobs('done', 1)
    assert invoke('buildings', 'pyramid').exit_code != 0
    assert batch.run.call_count == 1


def fire_score(monkeypatch: pytest.MonkeyPatch, plans: list[list[str]], *args: str):
    plan = MagicMock(side_effect=plans)
    monkeypatch.setattr(cli, 'plan_score', plan)
    monkeypatch.setattr(cli, 'stopped_by_hand', lambda cluster_id: None)
    cmd = ['stage', '--version', 'v2.0.0', '--coiled', *args, 'fire', 'score']
    return CliRunner().invoke(cli.app, cmd), plan


def test_score_rounds_resubmit_missing_parts(
    batch: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    kw = '{"score": {"region_ids": ["a", "b", "c"]}}'
    r, plan = fire_score(monkeypatch, [['a', 'b', 'c'], ['c'], []], '--overwrite', '--kwargs', kw)
    assert r.exit_code == 0, r.output
    # Overwrite only in round 1; region_ids in every plan.
    assert [(c.args[2], c.kwargs) for c in plan.call_args_list] == [
        (True, {'region_ids': ['a', 'b', 'c']}),
        (False, {'region_ids': ['a', 'b', 'c']}),
        (False, {'region_ids': ['a', 'b', 'c']}),
    ]
    runs = [c.kwargs for c in batch.run.call_args_list]
    assert [k['map_over_values'] for k in runs] == [['a', 'b', 'c'], ['c']]
    assert runs[0]['command'] == [
        'ocr',
        'stage',
        '--version',
        'v2.0.0',
        '--prefix',
        'output',
        'fire',
        'score',
    ]


def test_score_fails_after_rounds(batch: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    # Progress every round, but parts still missing after the last.
    plans = [['a', 'b', 'c', 'd'], ['b', 'c', 'd'], ['c', 'd'], ['d']]
    assert len(plans) == cli.SCORE_ROUNDS + 1
    r, _ = fire_score(monkeypatch, plans)
    assert r.exit_code != 0
    assert batch.run.call_count == cli.SCORE_ROUNDS
    assert "after 3 rounds; last logs: coiled batch logs 2. Missing: ['d']" in str(r.exception)


def test_score_stops_on_no_progress(batch: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    # Round 1 wrote nothing: no second VM per part.
    r, _ = fire_score(monkeypatch, [['a', 'b'], ['a', 'b']])
    assert r.exit_code != 0
    assert batch.run.call_count == 1
    assert 'wrote none of its 2 parts; logs: coiled batch logs 2' in str(r.exception)


def test_score_hand_stop_ends_run(batch: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    # A cluster stopped in the Coiled UI ends the run, not just its round.
    monkeypatch.setattr(cli, 'plan_score', MagicMock(side_effect=[['a', 'b'], ['b']]))
    monkeypatch.setattr(cli, 'stopped_by_hand', lambda cluster_id: 'me requested')
    r = CliRunner().invoke(cli.app, ['stage', '--version', 'v2.0.0', '--coiled', 'fire', 'score'])
    assert r.exit_code != 0
    assert batch.run.call_count == 1
    assert 'cluster stopped (me requested)' in str(r.exception)


def test_score_nothing_to_do(batch: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    r, _ = fire_score(monkeypatch, [[]])
    assert r.exit_code == 0, r.output
    batch.run.assert_not_called()


def test_score_dry_run_does_not_plan(batch: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    r, plan = fire_score(monkeypatch, [], '--dry-run')
    assert r.exit_code == 0, r.output
    plan.assert_not_called()
    batch.run.assert_not_called()


def test_coiled_forwards_overwrite(batch: MagicMock) -> None:
    assert invoke('buildings', '--overwrite').exit_code == 0
    assert batch.run.call_args.kwargs['command'][-1] == '--overwrite'


def test_overwrite_reaches_task(monkeypatch: pytest.MonkeyPatch) -> None:
    task = MagicMock()
    monkeypatch.setitem(STAGES, 'buildings', task)
    args = ['stage', '--version', 'v2.0.0', '--overwrite', 'flood', 'buildings']
    assert CliRunner().invoke(cli.app, args).exit_code == 0
    assert task.call_args.args[2] is True


def test_dry_run_orders_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    flood = dataclasses.replace(RISKS[Risk.FLOOD], score=lambda out: None)
    monkeypatch.setitem(RISKS, Risk.FLOOD, flood)
    r = CliRunner().invoke(
        cli.app,
        [
            'stage',
            '--version',
            'v2.0.0',
            '--dry-run',
            'flood',
            'regions',
            'buildings',
            'score',
            'regions',
        ],
    )
    # Reordered and deduped.
    assert r.exit_code == 0, r.output
    assert r.output.count('. ') == 3
    assert '1. score' in r.output and '2. buildings' in r.output and '3. regions' in r.output


RUN_FILE = """
risk = "flood"
version = "v2.0.0"
stages = ["buildings"]
coiled = true
[kwargs.buildings]
point_strategy = "ST_PointOnSurface"
[job.buildings]
vm_type = ["r8g.8xlarge"]
"""


def test_run_file_sends_kwargs_and_job(batch: MagicMock, tmp_path: Path) -> None:
    (f := tmp_path / 'run.toml').write_text(RUN_FILE)
    r = CliRunner().invoke(cli.app, ['run', str(f)])
    assert r.exit_code == 0, r.output
    kw = batch.run.call_args.kwargs
    assert '--kwargs' not in kw['command']
    assert kw['env'][cli.KWARGS_ENV] == json.dumps(
        {'buildings': {'point_strategy': 'ST_PointOnSurface'}}
    )
    assert kw['vm_type'] == ['r8g.8xlarge'] and kw['disk_size'] == '50 GiB'


def test_run_file_bad_kwarg_fails_before_submit(batch: MagicMock, tmp_path: Path) -> None:
    (f := tmp_path / 'run.toml').write_text(RUN_FILE.replace('point_strategy', 'point_strat'))
    r = CliRunner().invoke(cli.app, ['run', str(f)])
    assert r.exit_code != 0
    batch.run.assert_not_called()


def test_kwargs_json_reaches_task(monkeypatch: pytest.MonkeyPatch) -> None:
    # What a Coiled VM runs: the JSON kwargs arrive typed at the local task.
    task = MagicMock()
    monkeypatch.setitem(STAGES, 'buildings', task)
    kwargs = json.dumps({'buildings': {'point_strategy': 'ST_PointOnSurface'}})
    r = CliRunner().invoke(
        cli.app, ['stage', '--version', 'v2.0.0', 'flood', 'buildings', '--kwargs', kwargs]
    )
    assert r.exit_code == 0, r.output
    assert task.call_args.kwargs == {'point_strategy': PointSamplingStrategy.ON_SURFACE}


def test_coiled_forwards_storage_env_and_package_syncs(
    batch: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OCR_STORAGE_BUCKET', 'other-bucket')
    r = invoke('buildings')
    assert r.exit_code == 0, r.output
    kw = batch.run.call_args.kwargs
    assert kw['env'] == {
        'PYTHONUNBUFFERED': '1',
        'TMPDIR': '/scratch/batch',
        'OCR_STORAGE_BUCKET': 'other-bucket',
        'OCR_STORAGE_REGION': 'us-west-2',
    }
    assert kw['spot_policy'] == 'on-demand'
    # else Coiled installs the newest proj and no datum grids
    assert kw['package_sync_conda_extras'] == ['proj', 'proj-data']
    assert 'software' not in kw


def test_job_env_and_tag_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OCR_STORAGE_BUCKET', 'other-bucket')
    cfg = cli.job_config('buildings', {'env': {'FOO': '1'}, 'tag': {'Owner': 'x'}}, None)
    assert cfg['env'] == {
        'PYTHONUNBUFFERED': '1',
        'TMPDIR': '/scratch/batch',
        'OCR_STORAGE_BUCKET': 'other-bucket',
        'OCR_STORAGE_REGION': 'us-west-2',
        'FOO': '1',
    }
    assert cfg['tag'] == {'Project': 'OCR', 'Owner': 'x'}


def test_coiled_rejects_local_root(batch: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', '/tmp/x')
    assert invoke('buildings').exit_code != 0
    batch.run.assert_not_called()


def test_downstream_util_kwargs_reach_task(monkeypatch: pytest.MonkeyPatch) -> None:
    # tile_budgets belongs to write_buildings_lod, reached through write_buildings' **lod_kwargs.
    task = MagicMock()
    monkeypatch.setitem(STAGES, 'buildings', task)
    kwargs = json.dumps({'buildings': {'tile_budgets': {'4': 2000}}})
    args = ['stage', '--version', 'v2.0.0', 'flood', 'buildings', '--kwargs', kwargs]
    r = CliRunner().invoke(cli.app, args)
    assert r.exit_code == 0, r.output
    assert task.call_args.kwargs == {'tile_budgets': {4: 2000}}


def test_missing_input_fails_before_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, 'preflight', stages.preflight)
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    task = MagicMock()
    monkeypatch.setitem(STAGES, 'buildings', task)
    args = ['stage', '--version', 'v2.0.0', 'flood', 'buildings']
    r = CliRunner().invoke(cli.app, args)
    assert r.exit_code != 0 and 'run the flood ETL first' in r.output
    task.assert_not_called()

    # Once the flood store and TIGER blocks exist, buildings runs without score (pyramid skipped too).
    blocks = tmp_path / RegionLevel.BLOCK.key
    blocks.parent.mkdir(parents=True)
    blocks.touch()
    store = tmp_path / 'output/flood/v2.0.0/store.icechunk'
    store.mkdir(parents=True)
    (store / 'repo').touch()
    assert CliRunner().invoke(cli.app, args).exit_code == 0
    task.assert_called_once()


@pytest.mark.parametrize('dry_run', [False, True])
def test_unimplemented_stage_fails_before_submit(
    batch: MagicMock, dry_run: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Flood has no score yet: fail here, not after a Coiled VM boots.
    monkeypatch.setattr(cli, 'preflight', stages.preflight)
    r = invoke('score', *(['--dry-run'] if dry_run else []))
    assert r.exit_code != 0 and 'score: not implemented' in r.output
    batch.run.assert_not_called()


def test_bad_version_is_cli_error() -> None:
    r = CliRunner().invoke(
        cli.app, ['stage', '--version', 'nope', '--dry-run', 'flood', 'buildings']
    )
    # Not 'Invalid value for --version': on GitHub Actions typer colors option names, splitting it.
    assert r.exit_code == 2 and 'version must start with v' in r.output


def test_unknown_stage_in_kwargs_fails() -> None:
    args = ['stage', '--version', 'v2.0.0', 'flood', 'buildings', '--kwargs', '{"bogus": {}}']
    r = CliRunner().invoke(cli.app, args)
    assert r.exit_code != 0 and 'bogus' in r.output


def test_unknown_kwarg_lists_valid() -> None:
    kwargs = '{"buildings": {"nope": 1}}'
    args = ['stage', '--version', 'v2.0.0', '--dry-run', 'flood', 'buildings', '--kwargs', kwargs]
    r = CliRunner().invoke(cli.app, args)
    assert r.exit_code == 2 and 'valid buildings kwargs' in r.output and 'bbox' in r.output


@pytest.mark.parametrize('path', sorted(Path('runs').rglob('*.toml')), ids=str)
def test_run_files_dry_run(path: Path) -> None:
    # Schema, kwargs, unimplemented stages and job config; storage is stubbed (`inputs_present`).
    r = CliRunner().invoke(cli.app, ['run', str(path), '--dry-run'])
    assert r.exit_code == 0 and 'warning:' not in r.output, r.output


def test_stage_reads_kwargs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # What a Coiled job runs: kwargs arrive in env, not argv (`run_coiled`).
    monkeypatch.setenv(cli.KWARGS_ENV, '{"regions": {"levels": ["counties"]}}')
    args = ['stage', '--version', 'v2.0.0', '--dry-run', 'flood', 'regions']
    r = CliRunner().invoke(cli.app, args)
    assert r.exit_code == 0 and "kwargs={'levels': ['counties']}" in r.output, r.output


def test_bad_kwargs_json_is_cli_error() -> None:
    args = ['stage', '--version', 'v2.0.0', 'flood', 'buildings', '--kwargs', '{nope']
    assert CliRunner().invoke(cli.app, args).exit_code == 2


def test_bad_run_file_is_cli_error(tmp_path: Path) -> None:
    (f := tmp_path / 'run.toml').write_text(RUN_FILE + 'bogus = 1\n')
    r = CliRunner().invoke(cli.app, ['run', str(f), '--dry-run'])
    assert r.exit_code == 2 and 'bogus' in r.output


def test_unused_kwargs_warn(tmp_path: Path) -> None:
    (f := tmp_path / 'run.toml').write_text(
        RUN_FILE.replace('stages = ["buildings"]', 'stages = ["regions"]')
    )
    r = CliRunner().invoke(cli.app, ['run', str(f), '--dry-run'])
    assert r.exit_code == 0 and 'warning: kwargs for stages not run' in r.output


def test_run_stages_override(batch: MagicMock, tmp_path: Path) -> None:
    # A rerun of one stage keeps the file's job settings, drops its kwargs for other stages
    # without a warning, and takes --kwargs.
    (f := tmp_path / 'run.toml').write_text(RUN_FILE + '[job.regions]\nmax_retries = 3\n')
    kwargs = json.dumps({'regions': {'levels': ['counties']}})
    r = CliRunner().invoke(cli.app, ['run', str(f), '--stages', 'regions', '--kwargs', kwargs])
    assert r.exit_code == 0 and 'warning:' not in r.output, r.output
    kw = batch.run.call_args.kwargs
    assert kw['command'][7:] == ['regions'] and kw['env'][cli.KWARGS_ENV] == kwargs
    assert kw['max_retries'] == 3 and batch.run.call_count == 1


def test_run_kwargs_merge_per_stage(batch: MagicMock, tmp_path: Path) -> None:
    # --kwargs adds to the file's kwargs for a stage instead of replacing them.
    (f := tmp_path / 'run.toml').write_text(RUN_FILE)
    kwargs = json.dumps({'buildings': {'workers': 2}})
    assert CliRunner().invoke(cli.app, ['run', str(f), '--kwargs', kwargs]).exit_code == 0
    sent = json.loads(batch.run.call_args.kwargs['env'][cli.KWARGS_ENV])
    assert sent == {'buildings': {'point_strategy': 'ST_PointOnSurface', 'workers': 2}}


def test_vm_flag_beats_run_file(batch: MagicMock, tmp_path: Path) -> None:
    (f := tmp_path / 'run.toml').write_text(RUN_FILE)
    assert CliRunner().invoke(cli.app, ['run', str(f), '--vm', 'm8g.large']).exit_code == 0
    assert batch.run.call_args.kwargs['vm_type'] == ['m8g.large']


def test_run_file_prefix(batch: MagicMock, tmp_path: Path) -> None:
    (f := tmp_path / 'run.toml').write_text('prefix = "scratch/nrh"\n' + RUN_FILE)
    r = CliRunner().invoke(cli.app, ['run', str(f), '--dry-run'])
    assert r.exit_code == 0, r.output
    assert 's3://carbonplan-ocr/scratch/nrh/flood/v2.0.0' in r.output
    assert CliRunner().invoke(cli.app, ['run', str(f)]).exit_code == 0
    assert batch.run.call_args.kwargs['command'][4:6] == ['--prefix', 'scratch/nrh']


def test_bbox_needs_prerelease_version() -> None:
    kwargs = json.dumps({'buildings': {'bbox': [-122.5, 37.7, -122.3, 37.8]}})
    args = ['stage', '--dry-run', '--kwargs', kwargs, 'flood', 'buildings', '--version']
    r = CliRunner().invoke(cli.app, [*args, 'v2.0.0'])
    assert r.exit_code != 0 and 'v2.0.0-scratch' in r.output
    assert CliRunner().invoke(cli.app, [*args, 'v2.0.0-scratch']).exit_code == 0
    assert CliRunner().invoke(cli.app, [*args, 'v2.0.0', '--prefix', 'scratch/me']).exit_code == 0


def test_coiled_prints_job_and_logs(batch: MagicMock) -> None:
    r = invoke('buildings')
    assert 'job 1, cluster 2; logs: coiled batch logs 2' in r.output
