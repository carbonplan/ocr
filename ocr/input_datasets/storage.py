"""Storage utilities for Icechunk and S3 operations."""

import random
import time
from pathlib import Path

import boto3
import icechunk
import xarray as xr
from icechunk.xarray import to_icechunk
from upath import UPath

from ocr.console import console


class IcechunkReader:
    """Utility for reading datasets from Icechunk storage."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        region: str = 'us-west-2',
        *,
        debug: bool = False,
    ):
        """Initialize the Icechunk reader.

        Parameters
        ----------
        bucket : str
            S3 bucket name (for S3 storage) or local path (for filesystem storage).
        prefix : str
            S3 prefix or local subdirectory.
        region : str, default 'us-west-2'
            AWS region for S3 operations.
        debug : bool, default False
            Enable debug logging.
        """
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.debug = debug
        self._storage: icechunk.Storage | None = None
        self._repo: icechunk.Repository | None = None

    @property
    def storage(self) -> icechunk.Storage:
        """Get or create the Icechunk storage backend."""
        if self._storage is None:
            # Detect if this is S3 or local filesystem
            if self.bucket.startswith('s3://') or '/' not in self.bucket:
                # S3 storage
                if self.debug:
                    console.log(f'Creating S3 Icechunk storage: s3://{self.bucket}/{self.prefix}')
                self._storage = icechunk.s3_storage(
                    bucket=self.bucket, prefix=self.prefix, region=self.region
                )
            else:
                # Local filesystem storage
                full_path = Path(self.bucket) / self.prefix
                if self.debug:
                    console.log(f'Creating local Icechunk storage: {full_path}')
                self._storage = icechunk.local_filesystem_storage(path=str(full_path))

        return self._storage

    @property
    def repo(self) -> icechunk.Repository:
        """Get or create the Icechunk repository."""
        if self._repo is None:
            if self.debug:
                console.log('Opening Icechunk repository')
            self._repo = icechunk.Repository.open(self.storage)
        return self._repo


class IcechunkWriter:
    """Utility for writing datasets to Icechunk storage with error handling.

    This class encapsulates the common pattern of:
    1. Setting up S3 or local Icechunk storage
    2. Opening or creating a repository
    3. Writing data with conflict resolution
    4. Committing with automatic retries
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        region: str = 'us-west-2',
        *,
        dry_run: bool = False,
        max_retries: int = 5,
        debug: bool = False,
    ):
        """Initialize the Icechunk writer.

        Parameters
        ----------
        bucket : str
            S3 bucket name (for S3 storage) or local path (for filesystem storage).
        prefix : str
            S3 prefix or local subdirectory.
        region : str, default 'us-west-2'
            AWS region for S3 operations.
        dry_run : bool, default False
            If True, skip actual writes and only log operations.
        max_retries : int, default 5
            Maximum number of retries for conflict resolution.
        debug : bool, default False
            Enable debug logging.
        """
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.debug = debug
        self._storage: icechunk.Storage | None = None
        self._repo: icechunk.Repository | None = None

    @property
    def storage(self) -> icechunk.Storage:
        """Get or create the Icechunk storage backend."""
        if self._storage is None:
            # Detect if this is S3 or local filesystem
            if self.bucket.startswith('s3://') or '/' not in self.bucket:
                # S3 storage
                if self.debug:
                    console.log(f'Creating S3 Icechunk storage: s3://{self.bucket}/{self.prefix}')
                self._storage = icechunk.s3_storage(
                    bucket=self.bucket, prefix=self.prefix, region=self.region
                )
            else:
                # Local filesystem storage
                full_path = Path(self.bucket) / self.prefix
                if self.debug:
                    console.log(f'Creating local Icechunk storage: {full_path}')
                self._storage = icechunk.local_filesystem_storage(path=str(full_path))

        return self._storage

    @property
    def repo(self) -> icechunk.Repository:
        """Get or create the Icechunk repository."""
        if self._repo is None:
            if self.debug:
                console.log('Opening or creating Icechunk repository')
            self._repo = icechunk.Repository.open_or_create(self.storage)
        return self._repo

    def write(
        self,
        dataset: xr.Dataset,
        commit_message: str,
        *,
        branch: str = 'main',
        skip_housekeeping: bool = False,
    ) -> str:
        """Write a dataset to the Icechunk store with automatic conflict resolution.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to write to the store.
        commit_message : str
            Commit message for the snapshot.
        branch : str, default 'main'
            Branch to write to.
        skip_housekeeping : bool, default False
            If True, skip snapshot expiration and garbage collection after commit.

        Returns
        -------
        str
            Snapshot ID of the commit.
        """
        if self.dry_run:
            console.log(
                f'[DRY RUN] Would write dataset to Icechunk: '
                f's3://{self.bucket}/{self.prefix} (commit: {commit_message})'
            )
            return 'dry-run-snapshot-id'

        attempt = 0
        while attempt < self.max_retries:
            try:
                session = self.repo.writable_session(branch=branch)

                # Write the dataset
                if self.debug:
                    console.log('Writing dataset to Icechunk session...')
                to_icechunk(dataset, session, mode='w')

                # Rebase with BasicConflictSolver to resolve conflicts
                if self.debug:
                    console.log('Rebasing session with BasicConflictSolver (UseOurs)...')
                session.rebase(
                    icechunk.BasicConflictSolver(
                        on_chunk_conflict=icechunk.VersionSelection.UseOurs
                    )
                )

                # Commit the session
                snapshot_id = session.commit(commit_message)

                if self.debug:
                    console.log(
                        f'Successfully committed dataset (snapshot: {snapshot_id}): {commit_message}'
                    )

                # Housekeeping: expire old snapshots and garbage collect
                if not skip_housekeeping:
                    if self.debug:
                        console.log(
                            'Performing housekeeping (expire snapshots + garbage collect)...'
                        )

                    latest_commit = list(self.repo.ancestry(branch=branch))[0]
                    if self.debug:
                        console.log(f'Latest commit: {latest_commit}')

                    # Expire snapshots older than the latest commit
                    expired = self.repo.expire_snapshots(older_than=latest_commit.written_at)
                    console.log(
                        f'Expired {len(expired)} old snapshot(s) in s3://{self.bucket}/{self.prefix}'
                    )

                    # Garbage collect unreferenced chunks
                    gc_results = self.repo.garbage_collect(latest_commit.written_at)
                    if self.debug:
                        console.log(f'Garbage collection results: {gc_results}')

                return snapshot_id

            except Exception as exc:
                attempt += 1
                if attempt >= self.max_retries:
                    console.log(
                        f'[bold red]Failed to write after {self.max_retries} attempts: {exc}'
                    )
                    raise

                delay = random.uniform(3.0, 10.0)
                if self.debug:
                    console.log(f'Conflict detected (attempt {attempt}/{self.max_retries}): {exc}')
                    console.log(f'Retrying in {delay:.2f}s...')
                time.sleep(delay)

        raise RuntimeError('Unreachable code')


class S3Uploader:
    """Utility for uploading files to S3 with progress tracking."""

    def __init__(
        self, bucket: str, region: str = 'us-west-2', *, dry_run: bool = False, debug: bool = False
    ):
        """Initialize the S3 uploader.

        Parameters
        ----------
        bucket : str
            S3 bucket name.
        region : str, default 'us-west-2'
            AWS region.
        dry_run : bool, default False
            If True, skip actual uploads and only log operations.
        debug : bool, default False
            Enable debug logging.
        """
        self.bucket = bucket
        self.region = region
        self.dry_run = dry_run
        self.debug = debug
        self._s3_client = None

    @property
    def s3_client(self):
        """Get or create the boto3 S3 client."""
        if self._s3_client is None:
            self._s3_client = boto3.client('s3', region_name=self.region)
        return self._s3_client

    def upload_file(self, local_path: Path, s3_key: str) -> str:
        """Upload a single file to S3.

        Parameters
        ----------
        local_path : Path
            Local file path to upload.
        s3_key : str
            S3 key (path within bucket) for the uploaded file.

        Returns
        -------
        str
            S3 URI of the uploaded file.
        """
        s3_uri = f's3://{self.bucket}/{s3_key}'

        if self.dry_run:
            console.log(f'[DRY RUN] Would upload {local_path} -> {s3_uri}')
            return s3_uri

        if not local_path.exists():
            raise FileNotFoundError(f'Local file not found: {local_path}')

        file_size = local_path.stat().st_size
        console.log(f'Uploading {local_path.name} ({file_size / 1e6:.1f} MB) -> {s3_uri}')

        self.s3_client.upload_file(str(local_path), self.bucket, s3_key)

        if self.debug:
            console.log(f'Successfully uploaded to {s3_uri}')

        return s3_uri

    def upload_directory(self, local_dir: Path, s3_prefix: str, pattern: str = '*') -> list[str]:
        """Upload all files matching a pattern from a directory to S3.

        Parameters
        ----------
        local_dir : Path
            Local directory containing files to upload.
        s3_prefix : str
            S3 prefix for uploaded files.
        pattern : str, default '*'
            Glob pattern for matching files (e.g., '*.tif').

        Returns
        -------
        list[str]
            List of S3 URIs for uploaded files.
        """
        if not local_dir.exists():
            raise FileNotFoundError(f'Local directory not found: {local_dir}')

        files = sorted(local_dir.glob(pattern))
        if not files:
            console.log(f'No files matching pattern "{pattern}" found in {local_dir}')
            return []

        console.log(f'Found {len(files)} files to upload matching pattern: {pattern}')
        uploaded_uris = []

        for file_path in files:
            if file_path.is_file():
                s3_key = f'{s3_prefix}/{file_path.name}'
                uri = self.upload_file(file_path, s3_key)
                uploaded_uris.append(uri)

        console.log(f'Uploaded {len(uploaded_uris)} files to s3://{self.bucket}/{s3_prefix}/')
        return uploaded_uris


def get_s3_path(bucket: str, prefix: str) -> UPath:
    """Construct a UPath for S3 storage.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        S3 prefix (path within bucket).

    Returns
    -------
    UPath
        UPath object for the S3 location.
    """
    return UPath(f's3://{bucket}/{prefix}')
