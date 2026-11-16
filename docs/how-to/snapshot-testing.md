# Snapshot testing

OCR uses [syrupy](https://github.com/tophat/syrupy) for snapshot testing with custom support for xarray DataArrays and Datasets stored in zarr format.

## Overview

The `XarraySnapshotExtension` allows you to:

- Create and compare snapshots of xarray DataArrays and Datasets
- Store snapshots in S3 (default) or locally
- Handle large datasets efficiently using zarr format

## Basic usage

### S3 snapshots (default)

By default, snapshots are stored in S3 at `s3://carbonplan-ocr/integration-tests/snapshots/`. This is configured in `tests/fire/conftest.py` and can be overridden if needed. The fire model's snapshot tests are in `tests/fire/risks/test_fire_snapshots.py` (integration, `pixi run tests-integration`); the examples below use a hypothetical `tests/test_snapshot.py`.

```python
import numpy as np
import pytest
import xarray as xr
from ocr.risks.fire.testing import XarraySnapshotExtension


@pytest.fixture
def xarray_snapshot(snapshot):
    return snapshot.use_extension(XarraySnapshotExtension)


def test_my_data(xarray_snapshot):
    data = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6]]),
        dims=('x', 'y'),
    )
    assert xarray_snapshot == data
```

Run tests with:

```bash
# Create/update snapshots in S3
pixi run pytest tests/test_snapshot.py --snapshot-update

# Verify snapshots against S3
pixi run pytest tests/test_snapshot.py
```

### Local storage (override)

If you prefer to store snapshots locally during development, you can override the default:

```bash
# Create/update local snapshots
SNAPSHOT_STORAGE_PATH=tests/__snapshots__ pixi run pytest tests/test_snapshot.py --snapshot-update

# Verify local snapshots
SNAPSHOT_STORAGE_PATH=tests/__snapshots__ pixi run pytest tests/test_snapshot.py
```

## Configuration

The snapshot storage location is controlled by the `SNAPSHOT_STORAGE_PATH` environment variable:

- **Default**: `s3://carbonplan-ocr/integration-tests/snapshots/` (set in `tests/fire/conftest.py`)
- **Override**: Set `SNAPSHOT_STORAGE_PATH` to any local path or S3 URI

### Examples

```bash
# Use default S3 storage (no env var needed)
pixi run tests-integration

# Override with local storage
SNAPSHOT_STORAGE_PATH=tests/__snapshots__ pixi run tests-integration

# Override with different S3 bucket
SNAPSHOT_STORAGE_PATH=s3://my-bucket/snapshots pixi run tests-integration
```

## Snapshot organization

Snapshots are named based on the test file and function:

`{test_file}_{test_function}.zarr`

Example:

- Test file: `tests/fire/risks/test_fire_snapshots.py`
- Test function: `test_wind_adjustment`
- Snapshot name: `test_fire_snapshots_test_wind_adjustment.zarr`

## Managing S3 snapshots

### List snapshots

```bash
aws s3 ls s3://carbonplan-ocr/integration-tests/snapshots/ --recursive
```

### Delete old snapshots

```bash
# Delete a specific snapshot
aws s3 rm s3://carbonplan-ocr/integration-tests/snapshots/test_snapshot_test_example.zarr --recursive

# Delete all snapshots (careful!)
aws s3 rm s3://carbonplan-ocr/integration-tests/snapshots/ --recursive
```

## Example: complete test setup

```python
# tests/test_fire_risk.py
import numpy as np
import pytest
import xarray as xr
from ocr.risks.fire.testing import XarraySnapshotExtension


@pytest.fixture
def xarray_snapshot(snapshot):
    return snapshot.use_extension(XarraySnapshotExtension)


@pytest.mark.integration  # mark this test as integration
def test_fire_risk_calculation(xarray_snapshot):
    # Your data processing
    result = xr.Dataset(
        {
            'fire_risk': xr.DataArray(
                np.random.rand(100, 100),
                dims=('lat', 'lon'),
                coords={
                    'lat': np.linspace(30, 40, 100),
                    'lon': np.linspace(-120, -110, 100),
                },
            )
        }
    )

    # Snapshot assertion
    assert xarray_snapshot == result
```

Run with:

```bash
# Default S3 storage
pixi run pytest tests/test_fire_risk.py --snapshot-update
pixi run pytest tests/test_fire_risk.py

# Local storage for development
SNAPSHOT_STORAGE_PATH=tests/__snapshots__ pixi run pytest tests/test_fire_risk.py --snapshot-update
SNAPSHOT_STORAGE_PATH=tests/__snapshots__ pixi run pytest tests/test_fire_risk.py

# Custom S3 bucket
SNAPSHOT_STORAGE_PATH=s3://my-bucket/snapshots pixi run pytest tests/test_fire_risk.py --snapshot-update
SNAPSHOT_STORAGE_PATH=s3://my-bucket/snapshots pixi run pytest tests/test_fire_risk.py
```
