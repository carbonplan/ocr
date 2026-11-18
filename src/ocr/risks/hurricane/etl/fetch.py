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
import sys
import zipfile
from fnmatch import fnmatch
from typing import Annotated

import obstore
import requests
import typer
from obspec_utils.readers import BufferedStoreReader
from obstore.store import HTTPStore

from ocr.config import StorageConfig

RAW_PREFIX = 'input/hurricane/raw/chaz'
EXPLORE_PREFIX = 'ocr-explore/CHAZ'
METRICS = ('return_periods', 'exceedance_intensity')

# Dryad file ids for dataset version 408938 (doi:10.5061/dryad.qfttdz0vz).
FILES: dict[str, dict] = {
    'exceedance_intensity': {'id': 4471224, 'gb': 44.69},
    'return_periods': {'id': 4471225, 'gb': 13.75},
}
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
    token = os.environ.get('DRYAD_TOKEN') or sys.exit('set DRYAD_TOKEN')
    r = requests.get(
        DOWNLOAD.format(id=file_id),
        headers={'Authorization': f'Bearer {token}'},
        allow_redirects=False,
        timeout=60,
    )
    if r.is_redirect and 'Location' in r.headers:
        return r.headers['Location']
    r.raise_for_status()
    sys.exit(f'expected a redirect to presigned storage, got {r.status_code}')


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
    parts = member_path.lstrip('/').split('/')
    base = parts[-1]
    if '__MACOSX' in parts or base == '.DS_Store' or base.startswith('._'):
        return None
    if parts and parts[0] == name:
        parts = parts[1:]
    return '/'.join(parts) if parts and parts[-1] else None


@app.command()
def fetch(
    name: Annotated[str, typer.Argument(help=f'one of {list(FILES)}')],
    only: Annotated[
        str | None, typer.Option(help="glob on member path, e.g. '*raster.nc*'")
    ] = None,
    dry_run: bool = False,
) -> None:
    """Stream members of one Dryad zip into RAW_PREFIX/<name>/."""
    if name not in FILES:
        sys.exit(f'unknown: {name}; known: {list(FILES)}')
    file_id = FILES[name]['id']
    with _open_zip(file_id) as zf:
        plan = [
            (info.filename, rel, info.file_size)
            for info in zf.infolist()
            if not info.is_dir()
            and (only is None or fnmatch(info.filename, only))
            and (rel := _clean_rel(info.filename, name)) is not None
        ]
    store = StorageConfig().object_store()
    print(f'{len(plan)} files, {sum(s for *_, s in plan) / 1e9:.2f} GB -> {RAW_PREFIX}/{name}/')
    copied = skipped = 0
    for i, (member, rel, size) in enumerate(plan, 1):
        dst = f'{RAW_PREFIX}/{name}/{rel}'
        if _size(store, dst) == size:
            skipped += 1
            continue
        print(f'  [{i}/{len(plan)}] {rel} ({size / 1e6:.1f} MB)')
        if dry_run:
            continue
        # Fresh presigned URL per member: one URL can expire during a multi-GB stream.
        with _open_zip(file_id) as zf, zf.open(member) as src:
            obstore.put(store, dst, src, chunk_size=32 << 20)
        copied += 1
    print(f'done: copied={copied} skipped={skipped}' + (' (dry run)' if dry_run else ''))


if __name__ == '__main__':
    app()
