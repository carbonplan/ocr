"""Raw CHAZ netCDF transfer into input/hurricane/raw/chaz/ (bytes unchanged; encoding happens in chaz.py).

Dataset: "Global coastal wind hazard maps from the CHAZ tropical cyclone model",
doi:10.5061/dryad.qfttdz0vz (CC0). Ported from carbonplan/ocr branch Shane98c/chaz (fetch_chaz.py).

copy-raw: server-side copy of the files already fetched to ocr-explore/CHAZ/ (complete except the
exceedance_intensity raster: 3/108 per-GCM files).

fetch: stream members of a Dryad zip into RAW_PREFIX. Dryad's download endpoint 302s to a presigned,
range-readable S3 URL (valid 24 h), so the multi-GB zip is read with range requests, never downloaded.
A fresh URL is resolved per member. Members already present at the same size are skipped (resumable).
    export DRYAD_TOKEN=<10 h token from your Dryad account page>
    pixi run hurricane-fetch fetch exceedance_intensity --only '*raster.nc*' --dry-run
On Coiled: coiled batch run --vm-type m8g.large --arm --tag Project=OCR --secret-env DRYAD_TOKEN=$DRYAD_TOKEN -- python -m ocr.risks.hurricane.etl.fetch fetch exceedance_intensity --only '*raster.nc*'
"""

import os
import zipfile
from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import Annotated, Literal, get_args

import obstore
import requests
import typer
from obspec_utils.readers import BufferedStoreReader
from obstore.store import HTTPStore

from ocr.config import INPUT_PREFIX, StorageConfig

RAW_PREFIX = f'{INPUT_PREFIX}/hurricane/raw/chaz'
EXPLORE_PREFIX = 'ocr-explore/CHAZ'
Metric = Literal['return_periods', 'exceedance_intensity']
METRICS: tuple[Metric, ...] = get_args(Metric)

# Dryad file ids for dataset version 408938 (doi:10.5061/dryad.qfttdz0vz).
FILES: dict[Metric, int] = {
    'exceedance_intensity': 4471224,
    'return_periods': 4471225,
}  # 44.7 / 13.8 GB
DOWNLOAD = 'https://datadryad.org/api/v2/files/{id}/download'

app = typer.Typer(no_args_is_help=True)


def _size(store, key: str) -> int | None:
    try:
        return obstore.head(store, key)['size']
    except FileNotFoundError:
        return None


@app.command()
def copy_raw(dry_run: bool = False) -> None:
    """Copy ocr-explore/CHAZ/{metric}/{nc,raster.nc}/**.nc to RAW_PREFIX (csv/ skipped)."""
    store = StorageConfig().object_store()
    copied = skipped = 0
    for metric in METRICS:
        for batch in obstore.list(store, f'{EXPLORE_PREFIX}/{metric}/'):
            for meta in batch:
                src = meta['path']
                if not src.endswith('.nc') or '/csv/' in src:
                    continue
                dst = f'{RAW_PREFIX}/{src.removeprefix(f"{EXPLORE_PREFIX}/")}'
                if _size(store, dst) == meta['size']:
                    skipped += 1
                    continue
                print(f'  {meta["size"] / 1e6:8.1f} MB  {dst}')
                if not dry_run:
                    obstore.copy(store, src, dst)
                copied += 1
    print(f'{"would copy" if dry_run else "copied"}={copied} skipped={skipped}')


def _presigned_url(file_id: int) -> str:
    """Resolve the Dryad download to its presigned, range-readable S3 URL."""
    token = os.environ.get('DRYAD_TOKEN')
    if not token:
        raise RuntimeError('set DRYAD_TOKEN')
    r = requests.get(
        DOWNLOAD.format(id=file_id),
        headers={'Authorization': f'Bearer {token}'},
        allow_redirects=False,
        timeout=60,
    )
    if r.is_redirect and 'Location' in r.headers:
        return r.headers['Location']
    r.raise_for_status()
    raise RuntimeError(f'expected a redirect to presigned storage, got {r.status_code}')


class _PresignedStore:
    """GET-only view of one presigned URL: the signature rejects HEAD, so size comes from a ranged GET."""

    def __init__(self, url: str) -> None:
        self._store = HTTPStore.from_url(url)

    def get(self, path: str, **kw):
        return obstore.get(self._store, path, **kw)

    def get_range(self, path: str, *, start: int, length: int):
        return obstore.get_range(self._store, path, start=start, length=length)

    def head(self, path: str):
        return obstore.get(self._store, path, options={'range': (0, 1)}).meta


def _open_zip(file_id: int) -> zipfile.ZipFile:
    reader = BufferedStoreReader(_PresignedStore(_presigned_url(file_id)), '', buffer_size=8 << 20)
    return zipfile.ZipFile(reader)


def _clean_rel(member_path: str, name: str) -> str | None:
    """Zip member path -> path under RAW_PREFIX/<name>/, or None to skip macOS cruft."""
    parts = PurePosixPath(member_path.lstrip('/')).parts
    if '__MACOSX' in parts or parts[-1].startswith('.'):
        return None
    return '/'.join(parts[1:] if parts[0] == name else parts)


@app.command()
def fetch(
    name: Metric,
    only: Annotated[str, typer.Option(help="glob on member path, e.g. '*raster.nc*'")] = '*',
    dry_run: bool = False,
) -> None:
    """Stream members of one Dryad zip into RAW_PREFIX/<name>/."""
    file_id = FILES[name]
    with _open_zip(file_id) as zf:
        members = [i for i in zf.infolist() if not i.is_dir() and fnmatch(i.filename, only)]
    plan = [(m, rel) for m in members if (rel := _clean_rel(m.filename, name))]
    store = StorageConfig().object_store()
    gb = sum(m.file_size for m, _ in plan) / 1e9
    print(f'{len(plan)} files, {gb:.2f} GB -> {RAW_PREFIX}/{name}/')
    copied = skipped = 0
    for i, (member, rel) in enumerate(plan, 1):
        dst = f'{RAW_PREFIX}/{name}/{rel}'
        if _size(store, dst) == member.file_size:
            skipped += 1
            continue
        print(f'  [{i}/{len(plan)}] {rel} ({member.file_size / 1e6:.1f} MB)')
        if dry_run:
            continue
        # Fresh presigned URL per member: one URL can expire during a multi-GB stream.
        with _open_zip(file_id) as zf, zf.open(member) as src:
            obstore.put(store, dst, src, chunk_size=32 << 20)
        copied += 1
    print(f'done: copied={copied} skipped={skipped}' + (' (dry run)' if dry_run else ''))


if __name__ == '__main__':
    app()
