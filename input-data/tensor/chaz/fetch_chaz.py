# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
#   "fsspec",
#   "aiohttp",
#   "s3fs>=2024.0.0",
# ]
# ///
"""Stream CHAZ coastal wind-hazard files from Dryad straight into S3.

Dataset: "Global coastal wind hazard maps from the CHAZ tropical cyclone model"
doi:10.5061/dryad.qfttdz0vz  (CC0).

Dryad gates downloads behind a 10-hour bearer token, but the download endpoint
302s to a *presigned S3 URL in us-west-2 that supports range requests* (valid
24h). So we never download the multi-GB zip: we open it over HTTPS with fsspec
and copy members one at a time into
s3://carbonplan-ocr/ocr-explore/CHAZ/<name>/. No big disk, in-region,
resumable (skips members already present at the same size).

Layout matches what a colleague already used for return_periods: the leading
`<name>/` dir inside the zip is stripped, so e.g. zip member
`exceedance_intensity/raster.nc/ERA5/foo_raster.nc` lands at
`CHAZ/exceedance_intensity/raster.nc/ERA5/foo_raster.nc`. macOS cruft is dropped.

Designed to run on a small Coiled VM via `uv run` (deps above). Set the token:

    export DRYAD_TOKEN=<paste 10h token>   # or pass via .env / coiled --secret-env-file

Usage:
  uv run fetch_chaz.py list
  uv run fetch_chaz.py fetch return_periods     --only '*raster.nc' --dry-run
  uv run fetch_chaz.py fetch return_periods     --only '*raster.nc'
  uv run fetch_chaz.py fetch exceedance_intensity --only '*raster.nc'
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from fnmatch import fnmatch

import fsspec
import requests
import s3fs

BUCKET = 'carbonplan-ocr'  # durable, no auto-expiry
PREFIX = 'ocr-explore/CHAZ'  # match the layout a colleague used for return_periods

# Dryad file ids for dataset version 408938 (doi:10.5061/dryad.qfttdz0vz).
FILES: dict[str, dict] = {
    'exceedance_intensity': {'id': 4471224, 'gb': 44.69},
    'return_periods': {'id': 4471225, 'gb': 13.75},
}

DOWNLOAD = 'https://datadryad.org/api/v2/files/{id}/download'


def _token() -> str:
    tok = os.environ.get('DRYAD_TOKEN')
    if not tok:
        sys.exit('set DRYAD_TOKEN (10h token from your Dryad account page)')
    return tok


def _presigned_url(file_id: int) -> str:
    """Resolve the Dryad download to its presigned, range-readable S3 URL."""
    r = requests.get(
        DOWNLOAD.format(id=file_id),
        headers={'Authorization': f'Bearer {_token()}'},
        allow_redirects=False,
        timeout=60,
    )
    if r.status_code in (301, 302, 303, 307, 308) and 'Location' in r.headers:
        return r.headers['Location']
    r.raise_for_status()
    sys.exit(f'expected a redirect to presigned storage, got {r.status_code}')


def _clean_rel(member_path: str, name: str) -> str | None:
    """Zip member path -> S3-relative path under CHAZ/<name>/, or None to skip.

    Strips the leading `<name>/` dir (so paths sit directly under CHAZ/<name>/,
    matching the colleague's return_periods layout) and drops macOS cruft
    (__MACOSX/, .DS_Store, ._* resource forks).
    """
    parts = member_path.lstrip('/').split('/')
    base = parts[-1]
    if '__MACOSX' in parts or base == '.DS_Store' or base.startswith('._'):
        return None
    if parts and parts[0] == name:
        parts = parts[1:]
    return '/'.join(parts) if parts else None


def cmd_fetch(name: str, only: str | None, dry_run: bool) -> None:
    if name not in FILES:
        sys.exit(f'unknown: {name}; known: {list(FILES)}')

    url = _presigned_url(FILES[name]['id'])
    print(f'  opening zip over presigned URL ({FILES[name]["gb"]} GB compressed)')
    zfs = fsspec.filesystem('zip', fo=url, target_protocol='https')

    members = [p for p in zfs.find('/') if not p.endswith('/')]
    if only:
        members = [m for m in members if fnmatch(m.lstrip('/'), only)]

    plan = []  # (member_path, rel, size)
    for m in members:
        rel = _clean_rel(m, name)
        if rel is not None:
            plan.append((m, rel, zfs.info(m).get('size') or 0))

    total = sum(s for _, _, s in plan)
    base = f'{BUCKET}/{PREFIX}/{name}'
    print(
        f'  {len(plan)} files, {total / 1e9:.2f} GB -> s3://{base}/'
        + (f'  (filter {only!r})' if only else '')
    )

    if dry_run:
        for _, rel, size in plan[:30]:
            print(f'    {size / 1e6:8.1f} MB  {base}/{rel}')
        if len(plan) > 30:
            print(f'    ... +{len(plan) - 30} more')
        print('  (dry run — nothing written)')
        return

    s3 = s3fs.S3FileSystem()
    t0 = time.time()
    copied = skipped = 0
    moved = 0
    for i, (m, rel, size) in enumerate(plan, 1):
        dst = f'{base}/{rel}'
        if s3.exists(dst) and s3.info(dst).get('size') == size:
            skipped += 1
            continue
        print(f'  [{i}/{len(plan)}] {rel}  ({size / 1e6:.1f} MB)')
        with zfs.open(m, 'rb') as src, s3.open(dst, 'wb') as out:
            shutil.copyfileobj(src, out, length=32 * 1024 * 1024)
        copied += 1
        moved += size

    dt = time.time() - t0
    print(
        f'\ndone: copied={copied} skipped={skipped} '
        f'({moved / 1e9:.2f} GB in {dt:.0f}s, {moved / 1e6 / (dt + 1e-9):.1f} MB/s)'
    )
    print(f'  -> s3://{base}/')


def cmd_list(_: str | None = None) -> None:
    for name, info in FILES.items():
        print(f'{name:24s} id={info["id"]:>8}  {info["gb"]:6.2f} GB (zip)')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch')
    f.add_argument('name')
    f.add_argument('--only', help="glob filter on member path, e.g. '*raster.nc'")
    f.add_argument(
        '--dry-run', action='store_true', help='list planned S3 keys and sizes, write nothing'
    )
    sub.add_parser('list')
    args = p.parse_args()

    if args.cmd == 'list':
        cmd_list()
    else:
        cmd_fetch(args.name, args.only, args.dry_run)


if __name__ == '__main__':
    main()
