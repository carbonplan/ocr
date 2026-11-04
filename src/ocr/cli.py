"""`ocr run FILE` (a run file under runs/<risk>/) or `ocr stage [opts] <risk> <stage>...`:
run stages in order, in process or one Coiled batch job per stage.
"""

import json
import os
import time
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, TypedDict

import typer
from pydantic import BaseModel, ConfigDict, ValidationError

from ocr.config import OUTPUT_PREFIX, OutputPaths, Risk, StorageConfig
from ocr.stages import RISKS, STAGES, check_storage, stage_kwargs, unimplemented


class JobConfig(TypedDict, total=False):  # keys = coiled.batch.run kwargs
    vm_type: list[str]
    arm: bool
    disk_size: str
    max_workers: int
    max_retries: int
    forward_aws_credentials: bool
    software: str
    region: str
    spot_policy: str  # 'on-demand', 'spot' or 'spot_with_fallback'
    job_timeout: str  # e.g. '6h'; Coiled stops the VM after this
    tag: dict[str, str]
    env: dict[str, str]


DEFAULT: JobConfig = {
    'arm': True,
    'forward_aws_credentials': True,
    'tag': {'Project': 'OCR'},
    'env': {'PYTHONUNBUFFERED': '1'},  # stream logs live; piped stdout is block-buffered
}
# Driver env vars the VM needs to write where the driver reads. LOCAL_ROOT can't go (see `execute`).
FORWARD_ENV = ('OCR_STORAGE_BUCKET', 'OCR_STORAGE_REGION')
# `ocr stage --kwargs` from env: how Coiled jobs get their stage kwargs (`run_coiled`).
KWARGS_ENV = 'OCR_STAGE_KWARGS'
# Estimates from data size until CONUS timing/RAM runs; vm_type lists are first-available.
JOBS: dict[str, JobConfig] = {
    'etl': {'vm_type': ['r8g.xlarge'], 'disk_size': '50 GiB'},  # whole tif in RAM, ~10-15 GB peak
    'score': {'vm_type': ['r8g.xlarge']},
    'pyramid': {
        'vm_type': ['r8g.2xlarge'],
        'spot_policy': 'spot_with_fallback',
        'job_timeout': '2h',
    },
    # Long single-task jobs: a spot reclaim loses hours.
    # Flood CONUS on r8g.8xlarge (2026-11-03): 64 min, ~1/4 of it at the gp3 write cap, mostly 1-2 cores busy.
    # r8gd: local NVMe at /scratch (~1 GB/s write vs 133 MB/s); /tmp is a small EBS root, so TMPDIR.
    'buildings': {
        'vm_type': ['r8gd.4xlarge'],
        'disk_size': '50 GiB',  # root only: temp files go to NVMe
        'env': {'TMPDIR': '/scratch/batch'},
        'spot_policy': 'on-demand',
        'job_timeout': '8h',
    },
    'regions': {
        'vm_type': ['r8g.4xlarge'],
        'disk_size': '200 GiB',
        'spot_policy': 'on-demand',
        'job_timeout': '4h',
    },
}

POLL_SECONDS = 10

Stage = StrEnum('Stage', {n: n for n in STAGES})


class RunFile(BaseModel):
    """`ocr run FILE`: one deployment of one risk, committed under runs/<risk>/."""

    model_config = ConfigDict(extra='forbid')

    risk: Risk
    version: str
    stages: list[Stage]
    coiled: bool = False
    bucket: str | None = None  # over env OCR_STORAGE_BUCKET (`StorageConfig`)
    prefix: str = OUTPUT_PREFIX  # of outputs, `OutputPaths.prefix`
    kwargs: dict[Stage, dict[str, Any]] = {}  # per stage: keyword args of its util, `STAGE_KWARGS`
    job: dict[Stage, JobConfig] = {}  # per stage: over JOBS


app = typer.Typer(
    add_completion=False,
    help="Run a risk's pipeline stages. Guide: docs/how-to/data-pipeline.md",
)


def job_config(stage: str, job: JobConfig, vm: str | None) -> JobConfig:
    # Same region as the bucket. No `software`: Coiled package-syncs the local env.
    runtime: JobConfig = {'region': StorageConfig().region}
    if env := {k: v for k in FORWARD_ENV if (v := os.environ.get(k))}:
        runtime['env'] = env
    layers = [DEFAULT, runtime, JOBS.get(stage, {}), job, {'vm_type': [vm]} if vm else {}]
    cfg: JobConfig = {k: v for layer in layers for k, v in layer.items()}  # last wins
    for key in ('env', 'tag'):  # dicts merge key by key, so a run file's env keeps FORWARD_ENV
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


def run_coiled(
    cfg: RunFile, stage: str, kwargs: BaseModel | None, vm: str | None, overwrite: bool
) -> None:
    """One Coiled batch job running `ocr stage` for this stage; blocks until it ends."""
    import coiled

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
    job = job_config(stage, cfg.job.get(stage, {}), vm)
    if kwargs is not None:
        # In env, not argv: Coiled rebuilds the command as a shell string and strips JSON's quotes.
        dumped = json.dumps({stage: kwargs.model_dump(mode='json', exclude_unset=True)})
        job['env'] = {**job.get('env', {}), KWARGS_ENV: dumped}
    j = coiled.batch.run(command=command, name=f'ocr-{cfg.risk}-{stage}', **job)
    logs = f'coiled batch logs {j["cluster_id"]}'  # takes the cluster id, not the job id
    typer.echo(f'{cfg.risk} {stage}: job {j["job_id"]}, cluster {j["cluster_id"]}; logs: {logs}')
    try:
        while (state := job_state(coiled.batch.status(j['cluster_id']))) is None:
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        typer.echo(f'{cfg.risk} {stage}: cancelling, deleting cluster {j["cluster_id"]}')
        coiled.delete_cluster(cluster_id=j['cluster_id'])
        raise
    if state == 'error':
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
    data = {'risk': risk, 'version': version, 'prefix': prefix, 'stages': stages, 'coiled': coiled}
    execute(load({**data, 'kwargs': kwargs or {}}, '--kwargs'), vm, dry_run, overwrite)


def execute(cfg: RunFile, vm: str | None, dry_run: bool, overwrite: bool) -> None:
    """Run the requested stages one at a time, in `STAGES` order. Unrequested stages are
    assumed done (their outputs already in storage).
    """
    risk = cfg.risk
    if cfg.bucket:
        # Env, so every StorageConfig() (inputs too) and FORWARD_ENV on Coiled see it.
        os.environ['OCR_STORAGE_BUCKET'] = cfg.bucket
    storage = StorageConfig()
    try:
        out = OutputPaths(risk, cfg.version, cfg.prefix)
    except ValidationError as e:
        raise typer.BadParameter(str(e), param_hint='--version / --prefix') from e
    if cfg.coiled and storage.local_root:
        raise typer.BadParameter('Coiled jobs write to S3: unset OCR_STORAGE_LOCAL_ROOT')
    checked = {}
    for s, kw in cfg.kwargs.items():
        try:
            checked[s] = stage_kwargs(s, kw)
        except ValidationError as e:
            valid = ', '.join(type(stage_kwargs(s, {})).model_fields) or 'none'
            raise typer.BadParameter(f'{e}\nvalid {s} kwargs: {valid}', param_hint='kwargs') from e
    if 'buildings' in checked and 'bbox' in checked['buildings'].model_fields_set:
        if out.prefix == OUTPUT_PREFIX and not out.version.prerelease:
            msg = (
                f'bbox subsets buildings: use --prefix scratch/<you> or e.g. v{out.version}-scratch'
            )
            raise typer.BadParameter(msg, param_hint='--version / --prefix')
    ordered = [s for s in STAGES if s in cfg.stages]
    if unused := sorted(set(cfg.kwargs) - set(ordered)):
        typer.echo(f'warning: kwargs for stages not run, ignored: {unused}', err=True)
    if todo := unimplemented(ordered, RISKS[risk]):
        raise typer.BadParameter(f'{risk}: ' + '; '.join(todo), param_hint='stages')
    given = {s: kw.model_dump(exclude_unset=True) for s, kw in checked.items()}
    if problems := check_storage(ordered, RISKS[risk], out, overwrite, given):
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
            kw = checked[stage].model_dump(exclude_unset=True) if stage in checked else {}
            STAGES[stage](RISKS[risk], out, overwrite, **kw)
        mins = (time.monotonic() - t0) / 60
        typer.echo(f'[{i}/{len(ordered)}] {risk} {stage}: done in {mins:.1f} min')
