"""`ocr run` / `ocr stage`: run a risk's stages in order, in process or one Coiled job each."""

import json
import re
import time
import tomllib
from collections.abc import Callable
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypedDict

import typer
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from ocr.config import OUTPUT_PREFIX, OutputPaths, Risk, StorageConfig
from ocr.stages import RISKS, STAGES, plan_score, preflight, stage_kwargs


class JobConfig(TypedDict, total=False):  # keys = coiled.batch.run kwargs
    vm_type: list[str]
    scheduler_vm_type: list[str]
    arm: bool
    forward_aws_credentials: bool
    disk_size: str
    max_workers: int
    max_retries: int
    software: str
    region: str
    spot_policy: Literal['on-demand', 'spot', 'spot_with_fallback']
    job_timeout: str  # e.g. '6h'; Coiled stops the VM after this
    tag: dict[str, str]
    env: dict[str, str]
    package_sync_conda_extras: list[str]


DEFAULT: JobConfig = {
    'arm': True,
    'forward_aws_credentials': False,
    # Fire needs the pinned proj and proj-data (check_proj_grids); package sync skips both
    # unless they're conda extras.
    'package_sync_conda_extras': ['proj', 'proj-data'],
    'spot_policy': 'on-demand',
    'tag': {'Project': 'OCR'},
    'env': {'PYTHONUNBUFFERED': '1'},  # stream logs live; piped stdout is block-buffered
}
# `ocr stage --kwargs` from env: how Coiled jobs get their stage kwargs (`run_coiled`).
KWARGS_ENV = 'OCR_STAGE_KWARGS'
# Estimates from data size until CONUS timing/RAM runs; vm_type lists are first-available.
JOBS: dict[str, JobConfig] = {
    # One task per fire region (cli.run_score). No max_workers: Coiled starts 1 VM per task (up to
    # 1,000), so all ~405 regions run at once, as the old fire pipeline did. Commits spread out with compute
    # time; contended ones rebase, and disjoint regions never conflict. If AWS vCPU quota bites, set
    # max_workers in [job.score].
    # ponytail: job_timeout is a guess; set ~2x the slowest region once timed. It's per job, so with
    # max_workers it must also cover queued tasks. A timeout ends the round; the next one re-plans.
    'score': {
        # 16 GB, 2 vCPU: regions are all 6000x4500, inland y4_x20 peaks ~9 GB, mostly one core.
        # Several types, as one-type spot runs lose VMs to interruptions (100 of 402 in the first
        # CONUS try); placement score 8 (2026-11-03). 405 x 2 vCPU fits the 1,219 spot vCPU quota.
        'vm_type': ['r8g.large', 'r9g.large', 'r7g.large'],
        # Only multi-task jobs: a 1-task coiled batch job runs its task on the scheduler VM.
        'scheduler_vm_type': ['m7g.xlarge'],
        'spot_policy': 'spot_with_fallback',
        'job_timeout': '2h',
    },
    # Fire CONUS (2026-11-03): ~2.5 GB flat, ~2 of 4 cores; flood is smaller.
    'pyramid': {'vm_type': ['m8g.4xlarge'], 'job_timeout': '2h'},
    # Flood CONUS on r8g.8xlarge (2026-11-03): 64 min, ~1/4 of it at the gp3 write cap, mostly 1-2 cores busy.
    # r8gd: local NVMe at /scratch (~1 GB/s write vs 133 MB/s); /tmp is a small EBS root, so TMPDIR.
    'buildings': {
        'vm_type': ['r8gd.8xlarge'],
        'disk_size': '50 GiB',  # root only: temp files go to NVMe
        'env': {'TMPDIR': '/scratch/batch'},
        'job_timeout': '8h',
    },
    'regions': {
        'vm_type': ['r8g.4xlarge'],
        'disk_size': '200 GiB',
        'job_timeout': '4h',
    },
}

POLL_SECONDS = 10
# Score map jobs per run; each round re-plans from the store's commit history.
SCORE_ROUNDS = 3

Stage = StrEnum('Stage', {n: n for n in STAGES})


class RunFile(BaseModel):
    """`ocr run FILE`: one deployment of one risk, committed under runs/<risk>/."""

    model_config = ConfigDict(extra='forbid')

    risk: Risk
    version: str
    stages: list[Stage]
    coiled: bool = False
    prefix: str = OUTPUT_PREFIX  # of outputs, `OutputPaths.prefix`
    kwargs: dict[Stage, dict[str, Any]] = {}  # per stage: keyword args of its util, `STAGE_KWARGS`
    job: dict[Stage, JobConfig] = {}  # per stage: over JOBS

    @cached_property
    def out(self) -> OutputPaths:
        return OutputPaths(self.risk, self.version, self.prefix)

    @model_validator(mode='after')
    def _check_out(self) -> Self:
        self.out  # bad version or prefix fails at load
        return self


app = typer.Typer(
    add_completion=False,
    help="Run a risk's pipeline stages. Guide: docs/how-to/data-pipeline.md",
)


def job_config(stage: str, job: JobConfig, vm: str | None) -> JobConfig:
    # Same bucket and region as the driver. No `software`: Coiled package-syncs the local env.
    s = StorageConfig()
    runtime: JobConfig = {
        'region': s.region,
        'env': {'OCR_STORAGE_BUCKET': s.bucket, 'OCR_STORAGE_REGION': s.region},
    }
    layers = [DEFAULT, runtime, JOBS.get(stage, {}), job, {'vm_type': [vm]} if vm else {}]
    cfg: JobConfig = {k: v for layer in layers for k, v in layer.items()}  # last wins
    for key in ('env', 'tag'):  # dicts merge key by key, so a run file's env keeps the storage env
        if merged := {k: v for layer in layers for k, v in layer.get(key, {}).items()}:
            cfg[key] = merged
    return cfg


def job_state(jobs: list[dict]) -> str | None:
    """'done', 'error' or None (still running), from `coiled.batch.status` the way
    `coiled batch wait` reads it. coiled.batch.wait_for_job_done never returns on error.
    """
    tasks = [t for j in jobs for t in j['tasks']]
    if any(t['exit_code'] or t['state'] == 'error' for t in tasks):
        return 'error'
    # Before the cluster check: Coiled stops the cluster once every task is done.
    if tasks and all(t['state'] == 'done' for t in tasks):
        return 'done'
    # 'stopped' before every task is done: job_timeout or a lost VM; Coiled's own wait hangs here.
    if jobs and jobs[0]['cluster_state'] in ('error', 'stopped'):
        return 'error'
    return None


def job_finished(jobs: list[dict]) -> bool:
    """Every task done or failed, or the cluster gone (`coiled batch wait`'s rule). For score map
    jobs, where the commit history, not exit codes, says which parts are done.
    """
    tasks = [t for j in jobs for t in j['tasks']]
    if tasks and all(t['exit_code'] or t['state'] in ('done', 'error') for t in tasks):
        return True
    return bool(jobs) and jobs[0]['cluster_state'] in ('error', 'stopped')


def wait(j: dict, until: Callable[[list[dict]], Any]) -> Any:
    """Poll job `j` until `until(status)` is truthy and return it. Ctrl-C deletes the cluster."""
    import coiled

    try:
        while not (result := until(coiled.batch.status(j['cluster_id']))):
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        typer.echo(f'cancelling, deleting cluster {j["cluster_id"]}')
        coiled.delete_cluster(cluster_id=j['cluster_id'])
        raise
    return result


def stopped_by_hand(cluster_id: int) -> str | None:
    """Coiled's stop reason if someone stopped the cluster (UI or CLI), else None."""
    from coiled.v2.core import CloudV2

    with CloudV2() as cloud:
        reason = cloud.cluster_details(cluster_id)['current_state']['reason']
    return reason if reason.endswith(' requested') else None


def run_score(cfg: RunFile, kwargs: BaseModel | None, job: JobConfig, overwrite: bool) -> None:
    """Score fans out, one Coiled task per part. Each round plans on this machine (commit history),
    runs a map job over the missing parts and waits for every task; up to SCORE_ROUNDS.
    """
    import coiled

    spec, out = RISKS[cfg.risk], cfg.out
    kw = kwargs.model_dump(exclude_unset=True) if kwargs is not None else {}
    # No --kwargs or --overwrite: the driver planned and overwrote; a task overwriting would
    # delete the store once per task. Each task reads its part from COILED_BATCH_TASK_INPUT.
    command = [
        'ocr',
        'stage',
        '--version',
        cfg.version,
        '--prefix',
        cfg.prefix,
        cfg.risk.value,
        'score',
    ]
    prev: list[str] = []
    logs = None
    for attempt in range(1, SCORE_ROUNDS + 1):
        if not (todo := plan_score(spec, out, overwrite and attempt == 1, **kw)):
            return
        # A round that wrote nothing failed the same way everywhere (a bug, not lost VMs): stop
        # before starting another VM per part. A rerun resumes.
        if todo == prev:
            raise RuntimeError(
                f'score: round {attempt - 1} wrote none of its {len(todo)} parts; '
                f'logs: {logs}. Missing: {todo}'
            )
        prev = todo
        typer.echo(f'{cfg.risk} score: round {attempt}/{SCORE_ROUNDS}, {len(todo)} parts')
        j = coiled.batch.run(
            command=command,
            name=re.sub(r'[^A-Za-z0-9_-]', '-', f'ocr-{cfg.risk}-score-{cfg.version}-{attempt}'),
            map_over_values=todo,
            **job,
        )
        logs = f'coiled batch logs {j["cluster_id"]}'
        typer.echo(f'{cfg.risk} score: job {j["job_id"]}, cluster {j["cluster_id"]}; logs: {logs}')
        wait(j, job_finished)
        # job_finished counts a stopped cluster as a finished round; a hand stop means stop the run.
        if reason := stopped_by_hand(j['cluster_id']):
            raise RuntimeError(
                f'score: round {attempt} cluster stopped ({reason}); rerun to resume'
            )
    if missing := plan_score(spec, out, False, **kw):
        raise RuntimeError(
            f'score: {len(missing)} parts missing after {SCORE_ROUNDS} rounds; '
            f'last logs: {logs}. Missing: {missing}'
        )


def run_coiled(
    cfg: RunFile, stage: str, kwargs: BaseModel | None, vm: str | None, overwrite: bool
) -> None:
    """One Coiled batch job running `ocr stage` for this stage (score: `run_score`); blocks until it ends."""
    import coiled

    job = job_config(stage, cfg.job.get(stage, {}), vm)
    if stage == 'score':
        return run_score(cfg, kwargs, job, overwrite)
    # The VM runs this stage in process; the run file stays here, so kwargs go as JSON.
    command = [
        'ocr',
        'stage',
        '--version',
        cfg.version,
        '--prefix',
        cfg.prefix,
        cfg.risk.value,
        stage,
    ]
    if overwrite:
        command.append('--overwrite')
    if kwargs is not None:
        # In env, not argv: Coiled rebuilds the command as a shell string and strips JSON's quotes.
        dumped = json.dumps({stage: kwargs.model_dump(mode='json', exclude_unset=True)})
        job['env'] = {**job.get('env', {}), KWARGS_ENV: dumped}
    # Coiled cluster names allow only letters, digits, '-' and '_'.
    name = re.sub(r'[^A-Za-z0-9_-]', '-', f'ocr-{cfg.risk}-{stage}-{cfg.version}')
    j = coiled.batch.run(command=command, name=name, **job)
    logs = f'coiled batch logs {j["cluster_id"]}'  # takes the cluster id, not the job id
    typer.echo(f'{cfg.risk} {stage}: job {j["job_id"]}, cluster {j["cluster_id"]}; logs: {logs}')
    if wait(j, job_state) == 'error':
        raise RuntimeError(f'{stage}: coiled job {j["job_id"]} failed; logs: {logs}')


def load(data: dict[str, Any], hint: str) -> RunFile:
    try:
        return RunFile.model_validate(data)
    except ValidationError as e:
        raise typer.BadParameter(str(e), param_hint=hint) from e


Vm = Annotated[str | None, typer.Option(help='Override the Coiled vm_type of every stage run')]
DryRun = Annotated[bool, typer.Option(help='Print the plan, run nothing')]
Overwrite = Annotated[bool, typer.Option(help='Replace existing outputs of the stages run')]
Kwargs = Annotated[
    dict[str, Any] | None,
    typer.Option(parser=json.loads, metavar='JSON', help='{stage: {kwarg: value}}'),
]

RUN_EXAMPLES = """Examples:

ocr run runs/flood/v2.0.0.toml --dry-run
ocr run runs/flood/v2.0.0.toml --stages regions --kwargs '{"regions": {"levels": ["counties"]}}'
ocr run runs/flood/v2.0.0.toml --stages buildings --kwargs '{"buildings": {"polygons_from": 13}}'
ocr run runs/flood/v2.0.0.toml --stages pyramid --kwargs '{"pyramid": {"branch": "test", "levels": 8}}'
"""
STAGE_EXAMPLES = """Examples:

ocr stage --version v2.0.0 --dry-run flood buildings regions
ocr stage --version v2.0.0 --coiled --vm r8g.4xlarge flood pyramid
ocr stage --version v2.0.0-scratch --prefix scratch/me --coiled flood buildings
ocr stage --version v2.0.0 --kwargs '{"regions": {"levels": ["counties"]}}' flood regions
"""


@app.command(epilog=RUN_EXAMPLES)
def run(
    file: Path,
    stages: Annotated[
        list[Stage] | None, typer.Option(help="Run only these stages, with this file's settings")
    ] = None,
    kwargs: Kwargs = None,
    vm: Vm = None,
    dry_run: DryRun = False,
    overwrite: Overwrite = False,
) -> None:
    """Run the stages of a run file (TOML, e.g. runs/flood/v2.0.0.toml)."""
    data = tomllib.loads(file.read_text())
    if stages:
        data['stages'] = stages
        # The file's kwargs for stages left out are expected here, not a mistake.
        data['kwargs'] = {k: v for k, v in data.get('kwargs', {}).items() if k in stages}
    if kwargs:
        # Per stage, so a rerun keeps the file's other kwargs for that stage.
        file_kw = data.get('kwargs', {})
        data['kwargs'] = file_kw | {s: file_kw.get(s, {}) | kw for s, kw in kwargs.items()}
    execute(load(data, 'FILE'), vm, dry_run, overwrite)


@app.command(epilog=STAGE_EXAMPLES)
def stage(
    risk: Risk,
    stages: Annotated[list[Stage], typer.Argument()],
    version: Annotated[str, typer.Option(help='Output version, e.g. v2.0.0')],
    prefix: Annotated[str, typer.Option(help='Output prefix, e.g. scratch/<you>')] = OUTPUT_PREFIX,
    coiled: Annotated[bool, typer.Option(help='One Coiled batch job per stage')] = False,
    vm: Vm = None,
    dry_run: DryRun = False,
    overwrite: Overwrite = False,
    kwargs: Annotated[
        dict[str, Any] | None,
        typer.Option(
            parser=json.loads, metavar='JSON', envvar=KWARGS_ENV, help='{stage: {kwarg: value}}'
        ),
    ] = None,
) -> None:
    """Run stages given on the command line (also what each Coiled job runs)."""
    data = {
        'risk': risk,
        'version': version,
        'prefix': prefix,
        'stages': stages,
        'coiled': coiled,
        'kwargs': kwargs or {},
    }
    execute(load(data, '--version / --prefix / --kwargs'), vm, dry_run, overwrite)


def execute(cfg: RunFile, vm: str | None, dry_run: bool, overwrite: bool) -> None:
    """Run the requested stages one at a time, in `STAGES` order. Unrequested stages are
    assumed done (their outputs already in storage).
    """
    risk, out, storage = cfg.risk, cfg.out, StorageConfig()
    if cfg.coiled and storage.local_root:
        raise typer.BadParameter('Coiled jobs write to S3: unset OCR_STORAGE_LOCAL_ROOT')
    checked = {}
    for s, kw in cfg.kwargs.items():
        try:
            checked[s] = stage_kwargs(s, kw)
        except ValidationError as e:
            valid = ', '.join(type(stage_kwargs(s, {})).model_fields) or 'none'
            raise typer.BadParameter(f'{e}\nvalid {s} kwargs: {valid}', param_hint='kwargs') from e
    given = {s: kw.model_dump(exclude_unset=True) for s, kw in checked.items()}
    production = out.prefix == OUTPUT_PREFIX and not out.version.prerelease
    if 'bbox' in given.get('buildings', {}) and production:
        msg = f'bbox subsets buildings: use --prefix scratch/<you> or e.g. v{out.version}-scratch'
        raise typer.BadParameter(msg, param_hint='--version / --prefix')
    ordered = [s for s in STAGES if s in cfg.stages]
    if unused := sorted(set(cfg.kwargs) - set(ordered)):
        typer.echo(f'warning: kwargs for stages not run, ignored: {unused}', err=True)
    if problems := preflight(ordered, RISKS[risk], out, given):
        raise typer.BadParameter('\n'.join(problems), param_hint='stages')

    if dry_run:
        typer.echo(
            f'{risk} {storage.uri(out.root)}'
            + (' (coiled)' if cfg.coiled else '')
            + (' (overwrite)' if overwrite else '')
        )
        for i, stage in enumerate(ordered, 1):
            extra = f'  {job_config(stage, cfg.job.get(stage, {}), vm)}' if cfg.coiled else ''
            if stage in checked:
                extra += f'  kwargs={checked[stage].model_dump(mode="json", exclude_unset=True)}'
            typer.echo(f'  {i}. {stage}{extra}')
        return

    # ponytail: stages run back to back, so pyramid and buildings don't overlap on Coiled.
    # If wall time matters, start both jobs and poll them in one loop.
    for i, stage in enumerate(ordered, 1):
        typer.echo(f'{risk} {stage}: start')
        t0 = time.monotonic()
        if cfg.coiled:
            run_coiled(cfg, stage, checked.get(stage), vm, overwrite)
        else:
            STAGES[stage](RISKS[risk], out, overwrite, **given.get(stage, {}))
        mins = (time.monotonic() - t0) / 60
        typer.echo(f'[{i}/{len(ordered)}] {risk} {stage}: done in {mins:.1f} min')
