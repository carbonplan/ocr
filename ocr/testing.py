import os
import shutil
import typing

import geopandas as gpd
import pandas as pd
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
      Default: s3://carbonplan-scratch/snapshots (configured in tests/conftest.py)

    Examples:
        # Use default S3 storage (no env var needed)
        pytest tests/test_snapshot.py --snapshot-update

        # Override with local storage
        SNAPSHOT_STORAGE_PATH=tests/__snapshots__ pytest tests/

        # Override with different S3 bucket
        SNAPSHOT_STORAGE_PATH=s3://my-bucket/snapshots pytest tests/
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
        test_file = upath.UPath(test_location.filepath).stem

        if base_path:
            # Remote storage: use configured base path with test file subdirectory
            return str(upath.UPath(base_path) / test_file)
        # Local storage: use __snapshots__ dir next to test file
        return str(upath.UPath(test_location.filepath).parent / '__snapshots__')

    @classmethod
    def get_snapshot_name(cls, *, test_location: PyTestLocation, index: SnapshotIndex = 0) -> str:
        """Generate snapshot name based on test name.

        Sanitizes the test name to replace problematic characters (e.g., brackets from
        parametrized tests) with underscores for valid file paths.
        """
        test_name = test_location.testname
        # Replace brackets and other problematic characters with underscores
        sanitized_name = test_name.replace('[', '_').replace(']', '').replace('/', '_')
        if index == 0:
            return sanitized_name
        return f'{sanitized_name}.{index}'

    @classmethod
    def get_location(cls, *, test_location: PyTestLocation, index: SnapshotIndex = 0) -> str:
        """Get the full snapshot location path.

        Override to properly handle S3 paths using upath instead of os.path.join.
        """
        dirname = cls.dirname(test_location=test_location)
        basename = cls.get_snapshot_name(test_location=test_location, index=index)
        filename = f'{basename}.{cls.file_extension}'

        # Use upath for proper S3/remote path handling
        snapshot_path = upath.UPath(dirname) / filename
        return str(snapshot_path)

    @classmethod
    def _warn_on_snapshot_name(cls, *, snapshot_name: str, test_location: PyTestLocation) -> None:
        """Override to disable snapshot name validation warnings.

        The base class warns if snapshot name doesn't contain test name, but our
        naming convention intentionally uses just the test name for cleaner organization.
        """
        pass

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
        """Check if serialized data matches snapshot using approximate comparison.

        Uses assert_allclose instead of assert_equal to handle platform-specific
        numerical differences from OpenCV and scipy operations between macOS and Linux.
        """
        if snapshot_data is None:
            return False
        try:
            # Compute both datasets to ensure all lazy operations are materialized
            serialized = serialized_data.compute()
            snapshot = snapshot_data.compute()

            # Use allclose for numerical tolerance (handles float32 precision + platform differences)
            # rtol=1e-6 and atol=1e-8 are reasonable for float32 with accumulated operations
            xr.testing.assert_allclose(serialized, snapshot, rtol=1e-6, atol=1e-8)
            return True
        except (AssertionError, TypeError):
            # TypeError can occur if data types don't match (e.g., string vs numeric)
            # Fall back to exact comparison for non-numeric data
            try:
                xr.testing.assert_equal(serialized_data.compute(), snapshot_data.compute())
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

        if isinstance(data, xr.Dataset):
            data.to_zarr(str(snapshot_path), mode='w')

    def diff_lines(
        self, serialized_data: typing.Any, snapshot_data: typing.Any
    ) -> typing.Iterator[str]:
        """Generate diff lines for test output."""
        try:
            # Try approximate comparison first
            serialized = serialized_data.compute()
            snapshot = snapshot_data.compute()
            xr.testing.assert_allclose(serialized, snapshot, rtol=1e-6, atol=1e-8)
            return iter([])
        except (AssertionError, TypeError) as e:
            # Provide detailed diff including both approximate and exact comparison attempts
            diff_output = [
                'Snapshot comparison failed (using rtol=1e-6, atol=1e-8 for numerical tolerance)',
                '-------------------------------',
                'Snapshot:',
            ] + str(snapshot_data).split('\n')

            diff_output += [
                '-------------------------------',
                'Serialized:',
            ] + str(serialized_data).split('\n')

            diff_output += [
                '-------------------------------',
                'Error:',
            ] + str(e).split('\n')

            return iter(diff_output)


class GeoDataFrameSnapshotExtension(SingleFileSnapshotExtension):
    """Snapshot extension for GeoPandas GeoDataFrames stored as parquet.

    Supports both local and remote (S3) storage via environment variable configuration:
    - SNAPSHOT_STORAGE_PATH: Base path for snapshots (local or s3://bucket/path)
      Default: s3://carbonplan-scratch/snapshots (configured in tests/conftest.py)

    Examples:
        # Use default S3 storage (no env var needed)
        pytest tests/test_snapshot.py --snapshot-update

        # Override with local storage
        SNAPSHOT_STORAGE_PATH=tests/__snapshots__ pytest tests/

        # Override with different S3 bucket
        SNAPSHOT_STORAGE_PATH=s3://my-bucket/snapshots pytest tests/
    """

    file_extension = 'parquet'

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
        test_file = upath.UPath(test_location.filepath).stem

        if base_path:
            # Remote storage: use configured base path with test file subdirectory
            return str(upath.UPath(base_path) / test_file)
        # Local storage: use __snapshots__ dir next to test file
        return str(upath.UPath(test_location.filepath).parent / '__snapshots__')

    @classmethod
    def get_snapshot_name(cls, *, test_location: PyTestLocation, index: SnapshotIndex = 0) -> str:
        """Generate snapshot name based on test name.

        Sanitizes the test name to replace problematic characters (e.g., brackets from
        parametrized tests) with underscores for valid file paths.
        """
        test_name = test_location.testname
        # Replace brackets and other problematic characters with underscores
        sanitized_name = test_name.replace('[', '_').replace(']', '').replace('/', '_')
        if index == 0:
            return sanitized_name
        return f'{sanitized_name}.{index}'

    @classmethod
    def get_location(cls, *, test_location: PyTestLocation, index: SnapshotIndex = 0) -> str:
        """Get the full snapshot location path.

        Override to properly handle S3 paths using upath instead of os.path.join.
        """
        dirname = cls.dirname(test_location=test_location)
        basename = cls.get_snapshot_name(test_location=test_location, index=index)
        filename = f'{basename}.{cls.file_extension}'

        # Use upath for proper S3/remote path handling
        snapshot_path = upath.UPath(dirname) / filename
        return str(snapshot_path)

    @classmethod
    def _warn_on_snapshot_name(cls, *, snapshot_name: str, test_location: PyTestLocation) -> None:
        """Override to disable snapshot name validation warnings.

        The base class warns if snapshot name doesn't contain test name, but our
        naming convention intentionally uses just the test name for cleaner organization.
        """
        pass

    def serialize(self, data: SerializableData, **kwargs: typing.Any) -> typing.Any:
        """Validate that data is a GeoDataFrame. Returns the data unchanged."""
        if isinstance(data, gpd.GeoDataFrame):
            return data
        raise TypeError(f'Expected gpd.GeoDataFrame, got {type(data)}')

    def matches(
        self,
        *,
        serialized_data: typing.Any,
        snapshot_data: typing.Any,
    ) -> bool:
        """Check if serialized data matches snapshot using GeoDataFrame comparison."""
        if snapshot_data is None:
            return False
        try:
            # Compare DataFrames (excluding geometry for now, will compare separately)
            pd.testing.assert_frame_equal(
                serialized_data.drop(columns='geometry'),
                snapshot_data.drop(columns='geometry'),
                check_dtype=True,
                check_index_type=True,
                check_column_type=True,
            )
            # Compare geometries using GeoPandas method
            if not serialized_data.geometry.geom_equals(snapshot_data.geometry).all():
                return False
            # Compare CRS
            if serialized_data.crs != snapshot_data.crs:
                return False
            return True
        except (AssertionError, AttributeError, ValueError):
            return False

    def read_snapshot_data_from_location(
        self, *, snapshot_location: str, snapshot_name: str, session_id: str
    ) -> gpd.GeoDataFrame | None:
        """Read parquet snapshot from disk."""
        try:
            return gpd.read_parquet(snapshot_location)
        except (OSError, FileNotFoundError, Exception):
            return None

    @classmethod
    def write_snapshot_collection(cls, *, snapshot_collection: SnapshotCollection) -> None:
        """Write snapshot collection to parquet format (local or remote)."""
        filepath = snapshot_collection.location
        data = next(iter(snapshot_collection)).data

        snapshot_path = upath.UPath(str(filepath))

        # Remove existing parquet if it exists
        if snapshot_path.exists():
            # For S3/remote paths, use upath's filesystem
            if str(snapshot_path).startswith('s3://'):
                fs = snapshot_path.fs
                if fs.exists(str(snapshot_path)):
                    fs.rm(str(snapshot_path), recursive=False)
            else:
                # Use standard removal for local paths
                snapshot_path.unlink()

        data.to_parquet(
            str(snapshot_path),
            index=False,
            compression='zstd',
            geometry_encoding='WKB',
            write_covering_bbox=True,
            schema_version='1.1.0',
        )

    def diff_lines(
        self, serialized_data: typing.Any, snapshot_data: typing.Any
    ) -> typing.Iterator[str]:
        """Generate diff lines for test output."""
        try:
            # Try the comparison
            pd.testing.assert_frame_equal(
                serialized_data.drop(columns='geometry'),
                snapshot_data.drop(columns='geometry'),
            )
            if not serialized_data.geometry.geom_equals(snapshot_data.geometry).all():
                raise AssertionError('Geometries do not match')
            if serialized_data.crs != snapshot_data.crs:
                raise AssertionError(f'CRS mismatch: {serialized_data.crs} != {snapshot_data.crs}')
            return iter([])
        except AssertionError as e:
            # Generate informative diff
            return iter(
                ['Snapshot:']
                + [f'  Shape: {snapshot_data.shape}']
                + [f'  CRS: {snapshot_data.crs}']
                + [f'  Columns: {list(snapshot_data.columns)}']
                + snapshot_data.head().to_string().split('\n')
                + ['-------------------------------']
                + ['Serialized:']
                + [f'  Shape: {serialized_data.shape}']
                + [f'  CRS: {serialized_data.crs}']
                + [f'  Columns: {list(serialized_data.columns)}']
                + serialized_data.head().to_string().split('\n')
                + ['-------------------------------']
                + ['Error:']
                + str(e).split('\n')
            )
