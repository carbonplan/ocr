from pathlib import Path

import icechunk
import numpy as np
import odc.geo.xr  # noqa: F401
import pytest
import rioxarray  # noqa: F401
import typer
import xarray as xr
from typer.testing import CliRunner

from ocr.config import StorageConfig
from ocr.risks.flood.etl.flood import RAW_KEY, STORE_PREFIX, main, read_tif

NODATA = -9999.0


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StorageConfig:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    values = np.arange(48, dtype='float64').reshape(6, 8) / 48
    values[0, 0] = NODATA
    da = xr.DataArray(
        values,
        dims=('y', 'x'),
        coords={'y': 2_000_050.0 - 100 * np.arange(6), 'x': -2_000_050.0 + 100 * np.arange(8)},
        attrs={'STATISTICS_MEAN': float('nan'), 'AREA_OR_POINT': 'Area'},
    )
    da = da.rio.write_crs('EPSG:5070').rio.write_nodata(NODATA)
    raw = tmp_path / RAW_KEY
    raw.parent.mkdir(parents=True)
    da.rio.to_raster(raw)
    return StorageConfig()


def _run(*args: str) -> object:
    app = typer.Typer()
    app.command()(main)
    return CliRunner().invoke(app, list(args))


def test_read_tif(cfg: StorageConfig) -> None:
    fdp = read_tif(cfg)['fdp']
    assert fdp.dims == ('y', 'x') and fdp.dtype == np.float32
    assert np.isnan(fdp.values[0, 0]) and np.isfinite(fdp.values[1:]).all()
    assert not any(k.startswith('STATISTICS_') for k in fdp.attrs)
    assert fdp.attrs['AREA_OR_POINT'] == 'Area'
    assert read_tif(cfg).proj.crs.to_epsg() == 5070


def _open(cfg: StorageConfig) -> tuple[icechunk.Repository, xr.Dataset]:
    repo = icechunk.Repository.open(cfg.icechunk_storage(STORE_PREFIX))
    return repo, xr.open_zarr(repo.readonly_session('main').store, consolidated=False)


def test_main_roundtrip(cfg: StorageConfig) -> None:
    result = _run()
    assert result.exit_code == 0, result.output
    _, ds = _open(cfg)
    np.testing.assert_array_equal(ds['fdp'].values, read_tif(cfg)['fdp'].values)
    assert 'EPSG:5070' in str(ds.attrs.get('proj:code'))
    assert ds.odc.crs.epsg == 5070  # crs_wkt on spatial_ref survives the reopen


def test_overwrite_guard(cfg: StorageConfig) -> None:
    assert _run().exit_code == 0
    repo, _ = _open(cfg)
    first = repo.lookup_branch('main')
    assert isinstance(_run().exception, FileExistsError)
    assert _run('--overwrite').exit_code == 0
    assert repo.lookup_branch('main') != first
