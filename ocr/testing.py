import os
import shutil
import typing

import upath
import xarray as xr
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.location import PyTestLocation
from syrupy.types import SerializableData, SnapshotIndex


class XarraySnapshotExtension(SingleFileSnapshotExtension):
    """Snapshot extension for xarray DataArrays and Datasets stored as zarr.

    Supports both local and remote (S3) storage via environment variable configuration:
    - SNAPSHOT_STORAGE_PATH: Base path for snapshots (local or s3://bucket/path)

    Examples:
        # Local storage (default)
        SNAPSHOT_STORAGE_PATH=tests/__snapshots__

        # S3 storage
        SNAPSHOT_STORAGE_PATH=s3://my-bucket/snapshots
    """

    file_extension = 'zarr'

    @classmethod
    def _get_base_snapshot_path(cls) -> upath.UPath | None:
        """Get the base snapshot storage path from environment or use default local path."""
        storage_path = os.environ.get('SNAPSHOT_STORAGE_PATH')
        if storage_path:
            return upath.UPath(storage_path)
        return None

    @classmethod
    def dirname(cls, *, test_location: PyTestLocation) -> str:
        """Return the directory for storing snapshots."""
        base_path = cls._get_base_snapshot_path()
        if base_path:
            # Remote storage: use configured base path
            return str(base_path)
        # Local storage: use __snapshots__ dir next to test file
        return str(upath.UPath(test_location.filepath).parent / '__snapshots__')

    @classmethod
    def get_snapshot_name(cls, *, test_location: PyTestLocation, index: SnapshotIndex = 0) -> str:
        """Generate snapshot name based on test location and test name."""
        test_file = upath.UPath(test_location.filepath).stem
        test_name = test_location.testname
        if index == 0:
            return f'{test_file}_{test_name}'
        return f'{test_file}_{test_name}.{index}'

    def serialize(self, data: SerializableData, **kwargs: typing.Any) -> typing.Any:
        """Convert DataArray to Dataset for consistent zarr storage. Returns the data unchanged."""
        if isinstance(data, xr.DataArray):
            return data.to_dataset(name='data')
        elif isinstance(data, xr.Dataset):
            return data
        raise TypeError(f'Expected xr.DataArray or xr.Dataset, got {type(data)}')

    def matches(
        self,
        *,
        serialized_data: typing.Any,
        snapshot_data: typing.Any,
    ) -> bool:
        """Check if serialized data matches snapshot using xarray.testing.assert_equal."""
        if snapshot_data is None:
            return False
        try:
            xr.testing.assert_equal(serialized_data, snapshot_data)
            return True
        except AssertionError:
            return False

    def read_snapshot_data_from_location(
        self, *, snapshot_location: str, snapshot_name: str, session_id: str
    ) -> xr.Dataset | None:
        """Read zarr snapshot from disk."""
        try:
            return xr.open_dataset(snapshot_location, engine='zarr', chunks={})
        except (OSError, FileNotFoundError, IsADirectoryError):
            return None

    @classmethod
    def write_snapshot_collection(cls, *, snapshot_collection: SnapshotCollection) -> None:
        """Write snapshot collection to zarr format (local or remote)."""
        filepath = snapshot_collection.location
        data = next(iter(snapshot_collection)).data

        snapshot_path = upath.UPath(filepath)

        # Remove existing zarr if it exists
        if snapshot_path.exists():
            # For S3/remote paths, use upath's filesystem
            if str(snapshot_path).startswith('s3://'):
                fs = snapshot_path.fs
                if fs.exists(str(snapshot_path)):
                    fs.rm(str(snapshot_path), recursive=True)
            else:
                # Use standard shutil for local paths
                shutil.rmtree(snapshot_path)

        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, xr.Dataset):
            data.to_zarr(str(snapshot_path), mode='w')

    def diff_lines(
        self, serialized_data: typing.Any, snapshot_data: typing.Any
    ) -> typing.Iterator[str]:
        """Generate diff lines for test output."""
        try:
            xr.testing.assert_equal(serialized_data, snapshot_data)
            return iter([])
        except AssertionError as e:
            return iter(
                ['Snapshot:']
                + str(snapshot_data).split('\n')
                + ['-------------------------------']
                + ['Serialized:']
                + str(serialized_data).split('\n')
                + str(e).split('\n')
            )
