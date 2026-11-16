from pathlib import Path

import pytest

from ocr.config import OutputPaths, Risk, StorageConfig


def test_output_paths():
    out = OutputPaths('hurricane', 'v2.1.0')
    assert out.risk is Risk.HURRICANE
    assert out.store == 'output/hurricane/v2.1.0/store.icechunk'
    assert out.buildings == 'output/hurricane/v2.1.0/vector/buildings.geoparquet'
    assert out.regions == 'output/hurricane/v2.1.0/vector/regions'


def test_output_paths_resolve(tmp_path: Path):
    out = OutputPaths(Risk.FLOOD, 'v2.0.0')
    s3 = StorageConfig(bucket='other-bucket', local_root=None)
    assert (
        s3.uri(out.buildings) == 's3://other-bucket/output/flood/v2.0.0/vector/buildings.geoparquet'
    )
    local = StorageConfig(local_root=tmp_path)
    assert local.uri(out.store) == str(tmp_path / 'output/flood/v2.0.0/store.icechunk')
    assert local.uri(out.regions) == str(tmp_path / 'output/flood/v2.0.0/vector/regions')


@pytest.mark.parametrize('version', ['2.0.0', 'v2', 'v2.0', 'latest'])
def test_bad_version(version: str):
    with pytest.raises(ValueError):
        OutputPaths(Risk.FIRE, version)


@pytest.mark.parametrize('prefix', ['/abs', '../x', 'scratch/../output', './x', 'a//b', 'a/', ''])
def test_bad_prefix(prefix: str):
    with pytest.raises(ValueError, match='prefix'):
        OutputPaths(Risk.FIRE, 'v2.0.0', prefix)


@pytest.mark.parametrize('prefix', ['output', 'scratch/nrh', 'scratch/v2.0.0-dev'])
def test_good_prefix(prefix: str):
    assert OutputPaths(Risk.FIRE, 'v2.0.0', prefix).root.startswith(prefix)


def test_bad_risk():
    with pytest.raises(ValueError):
        OutputPaths('avalanche', 'v2.0.0')
