import zipfile
from pathlib import Path

import icechunk
import numpy as np
import odc.geo.xr  # noqa: F401
import pytest
import typer
import xarray as xr
from typer.testing import CliRunner

from ocr.config import INPUT_PREFIX, StorageConfig
from ocr.risks.hurricane.etl import chaz, fetch as fetch_mod
from ocr.risks.hurricane.etl.chaz import STORE, era5_key, gcm_key, run
from ocr.risks.hurricane.etl.fetch import EXPLORE_PREFIX, METRICS, RAW_PREFIX, _clean_rel, copy_raw

STEP = 1 / 12
# Coastal points on a 1/12 deg lattice anchored half a cell off the raw minimum (the ERA5 case).
LAT = (np.arange(4) + 0.5) * STEP + 10
LON = (np.arange(5) + 0.5) * STEP - 80
PRESENT = [
    ('CESM2', 'base', 'CRH'),
    ('MIROC6', 'fut1', 'SD'),
]  # ssp245 only; every other file missing


def _points(seed: int, metric: str = 'return_periods', lat: np.ndarray = LAT) -> xr.Dataset:
    lat, lon = (a.ravel() for a in np.meshgrid(lat, LON, indexing='ij'))
    keep = np.arange(lat.size) % 3 != 0  # sparse, like coastal centroids
    rng = np.random.default_rng(seed)
    bands = {b: ('points', rng.random(keep.sum())) for b in chaz.BANDS[metric]}
    return xr.Dataset(bands, coords={'lat': ('points', lat[keep]), 'lon': ('points', lon[keep])})


def _raster(seed: int, metric: str = 'return_periods') -> xr.Dataset:
    """Grid values on LAT x LON (what the store should hold)."""
    rng = np.random.default_rng(seed)
    shape = (LAT.size, LON.size)
    bands = {b: (('lat', 'lon'), rng.random(shape)) for b in chaz.BANDS[metric]}
    return xr.Dataset(bands, coords={'lat': LAT, 'lon': LON})


def _raw_raster(ds: xr.Dataset) -> xr.Dataset:
    """Raw raster layout: float noise on coords and one extra lon column (the repeated meridian)."""
    extra = ds.isel(lon=[0]).assign_coords(lon=[LON[-1] + STEP])
    ds = xr.concat([ds, extra], 'lon')
    return ds.assign_coords(lat=ds.lat + 1e-9, lon=ds.lon - 1e-9)


def _era5_raster(metric: str) -> xr.Dataset:
    """Raw ERA5 raster layout: bands prefixed thr_thr_* / rp_rp_*, plus per-cell point bookkeeping."""
    ds = _raw_raster(_raster(9, metric)).assign_coords(lat=lambda d: d.lat - 1)
    first = chaz.BANDS[metric][0]
    pre = first.split('_')[0]
    ds = ds.rename({b: f'{pre}_{b}' for b in chaz.BANDS[metric]})
    zero = ds[f'{pre}_{first}'] * 0
    return ds.assign({f'{pre}_points': zero, f'{pre}_lat': zero})


def _era5_points(metric: str = 'return_periods') -> xr.Dataset:
    return _points(8, metric, lat=LAT - 1)


def _write(ds: xr.Dataset, root: Path, key: str) -> None:
    (root / key).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root / key, engine='h5netcdf')


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StorageConfig:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    grids = {(rep, 'gcm'): (LAT, LON) for rep in ('points', 'raster')}
    grids |= {(rep, 'era5'): (LAT - 1, LON) for rep in ('points', 'raster')}
    monkeypatch.setattr(chaz, 'GRIDS', grids)
    for metric in METRICS:
        for seed, (gcm, period, variant) in enumerate(PRESENT):
            key = gcm_key(metric, gcm, 'ssp245', period, variant, 'points')
            _write(_points(seed, metric), tmp_path, key)
            key = gcm_key(metric, gcm, 'ssp245', period, variant, 'raster')
            _write(_raw_raster(_raster(seed, metric)), tmp_path, key)
        _write(_era5_raster(metric), tmp_path, era5_key(metric, 'raster'))
        _write(_era5_points(metric), tmp_path, era5_key(metric, 'points'))
    return StorageConfig()


def _open(cfg: StorageConfig, group: str) -> xr.Dataset:
    repo = icechunk.Repository.open(cfg.icechunk_storage(f'{INPUT_PREFIX}/{STORE}'))
    return xr.open_zarr(repo.readonly_session('main').store, group=group, consolidated=False)


def _messages(cfg: StorageConfig) -> list[str]:
    repo = icechunk.Repository.open(cfg.icechunk_storage(f'{INPUT_PREFIX}/{STORE}'))
    return [s.message for s in repo.ancestry(branch='main')]


@pytest.mark.parametrize('rep', ['points', 'raster'])
def test_roundtrip(cfg: StorageConfig, rep: str) -> None:
    run('return_periods', rep)
    ds = _open(cfg, chaz.group_name('return_periods', rep))
    assert ds['thr_33'].dims == chaz.DIMS and ds['thr_33'].dtype == np.float32
    assert list(ds['gcm'].values) == chaz.GCMS and list(ds['scenario'].values) == chaz.SCENARIOS
    np.testing.assert_allclose(ds['lat'], LAT)
    np.testing.assert_allclose(ds['lon'], LON)
    for seed, (gcm, period, variant) in enumerate(PRESENT):
        got = ds['thr_33'].sel(scenario='ssp245', period=period, variant=variant, gcm=gcm).values
        if rep == 'raster':
            np.testing.assert_allclose(got, _raster(seed)['thr_33'].values.astype('float32'))
        else:
            src = _points(seed)
            assert np.isfinite(got).sum() == src.sizes['points']
            np.testing.assert_allclose(
                np.sort(got[np.isfinite(got)]), np.sort(src['thr_33'].values), rtol=1e-6
            )
    # missing files stay NaN at their fixed index
    assert np.isnan(ds['thr_33'].sel(gcm='CNRM-CM6-1')).all()
    assert np.isnan(ds['thr_50'].sel(scenario='ssp585')).all()
    assert ds.attrs['bands'] == ['thr_33', 'thr_50']
    assert ds.attrs['source_doi'] == 'https://doi.org/10.5061/dryad.qfttdz0vz'
    assert ds['thr_50'].attrs['units'] == 'years' and '50 m/s' in ds['thr_50'].attrs['long_name']
    assert 'EPSG:4326' in str(ds.attrs.get('proj:code'))
    assert ds.odc.crs.epsg == 4326  # crs_wkt on spatial_ref survives the reopen
    assert _messages(cfg)[0] == chaz.group_name('return_periods', rep)


def test_snap_and_duplicates(cfg: StorageConfig, tmp_path: Path) -> None:
    pts = xr.Dataset(
        {'thr_33': ('points', [5.0, 1.0, 2.0]), 'thr_50': ('points', [0.0, 0.0, 0.0])},
        coords={
            'lat': (
                'points',
                [LAT[0] + 0.01, LAT[1], LAT[1] + 0.001],
            ),  # off-phase; two in one cell
            'lon': ('points', [LON[0], LON[1], LON[1]]),
        },
    )
    _write(pts, tmp_path, 'pts.nc')
    got = chaz.load(cfg, 'pts.nc', 'return_periods', 'points', 'gcm')['thr_33']
    assert got.isel(lat=0, lon=0) == 5.0  # off-phase point snapped to the nearest node
    assert got.isel(lat=1, lon=1) == 2.0  # shared cell: last value kept
    assert int(np.isfinite(got).sum()) == 2


def test_exists_guard(cfg: StorageConfig) -> None:
    run('return_periods')
    with pytest.raises(FileExistsError, match='--overwrite'):
        run('return_periods')


def test_overwrite_only_rebuilds_group(cfg: StorageConfig) -> None:
    run('return_periods', 'points')
    run('return_periods', 'raster')
    before = _open(cfg, 'return_periods_raster')['thr_33'].values
    run('return_periods', 'points', overwrite=True)
    assert _messages(cfg)[:3] == ['return_periods', 'return_periods_raster', 'return_periods']
    np.testing.assert_array_equal(_open(cfg, 'return_periods_raster')['thr_33'].values, before)
    assert np.isfinite(_open(cfg, 'return_periods')['thr_33'].sel(scenario='ssp245')).any()


@pytest.mark.parametrize('metric', METRICS)
@pytest.mark.parametrize('rep', ['points', 'raster'])
def test_era5_roundtrip(cfg: StorageConfig, rep: str, metric: str) -> None:
    run(metric, rep)
    run(metric, rep, source='era5')
    ds = _open(cfg, chaz.group_name(metric, rep, era5=True))
    band = chaz.BANDS[metric][0]
    assert ds[band].dims == ('lat', 'lon') and ds[band].dtype == np.float32
    assert set(ds.data_vars) == set(chaz.BANDS[metric]) and ds.attrs['forcing'] == 'ERA5'
    assert ds[band].attrs == chaz.band_attrs(metric, band)
    np.testing.assert_allclose(ds['lat'], LAT - 1)  # own grid, not the GCM one
    if rep == 'raster':
        np.testing.assert_allclose(ds[band], _raster(9, metric)[band].values.astype('float32'))
    else:
        got, src = ds[band].values, _era5_points(metric)
        assert np.isfinite(got).sum() == src.sizes['points']
        np.testing.assert_allclose(
            np.sort(got[np.isfinite(got)]), np.sort(src[band].values), rtol=1e-6
        )
    gcm = _open(cfg, chaz.group_name(metric, rep))
    np.testing.assert_allclose(gcm['lat'], LAT)
    assert _messages(cfg)[0] == chaz.group_name(metric, rep, era5=True)


def test_era5_missing(cfg: StorageConfig, tmp_path: Path) -> None:
    (tmp_path / era5_key('exceedance_intensity', 'points')).unlink()
    with pytest.raises(FileNotFoundError):
        run('exceedance_intensity', source='era5')


def test_raster_grid_mismatch(cfg: StorageConfig, tmp_path: Path) -> None:
    gcm, period, variant = PRESENT[1]
    key = gcm_key('return_periods', gcm, 'ssp245', period, variant, 'raster')
    _write(_raster(1).assign_coords(lat=LAT + 1), tmp_path, key)
    with pytest.raises(KeyError):
        run('return_periods', 'raster')


def test_prefix(cfg: StorageConfig, tmp_path: Path) -> None:
    run('return_periods', prefix='scratch/me')
    assert (tmp_path / 'scratch/me' / STORE).is_dir()
    assert not (tmp_path / 'input' / STORE).exists()


def test_cli_era5(cfg: StorageConfig) -> None:
    app = typer.Typer()
    app.command()(chaz.run)
    res = CliRunner().invoke(app, ['--metric', 'return_periods', '--source', 'era5'])
    assert res.exit_code == 0, res.output
    assert _messages(cfg)[0] == 'return_periods_era5'
    assert CliRunner().invoke(app, ['--metric', 'nope']).exit_code != 0


def test_clean_rel() -> None:
    assert _clean_rel('return_periods/nc/ERA5/a.nc', 'return_periods') == 'nc/ERA5/a.nc'
    assert _clean_rel('/nc/a.nc', 'return_periods') == 'nc/a.nc'
    assert _clean_rel('__MACOSX/nc/._a.nc', 'return_periods') is None
    assert _clean_rel('return_periods/nc/.DS_Store', 'return_periods') is None


def test_copy_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    for rel in (
        'return_periods/nc/ERA5/a.nc',
        'return_periods/csv/b.csv',
        'exceedance_intensity/raster.nc/c.nc',
    ):
        (tmp_path / EXPLORE_PREFIX / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / EXPLORE_PREFIX / rel).write_bytes(b'x' * 10)
    copy_raw(dry_run=True)
    assert not (tmp_path / RAW_PREFIX).exists()
    copy_raw()
    got = sorted(
        p.relative_to(tmp_path / RAW_PREFIX).as_posix()
        for p in (tmp_path / RAW_PREFIX).rglob('*')
        if p.is_file()
    )
    assert got == ['exceedance_intensity/raster.nc/c.nc', 'return_periods/nc/ERA5/a.nc']


def test_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path / 'bucket'))
    zpath = tmp_path / 'rp.zip'
    with zipfile.ZipFile(zpath, 'w') as zf:
        zf.writestr('return_periods/nc/ERA5/a.nc', b'a' * 10)
        zf.writestr('return_periods/raster.nc/ERA5/b.nc', b'b' * 20)
        zf.writestr('__MACOSX/return_periods/._a.nc', b'junk')
    monkeypatch.setattr(fetch_mod, '_open_zip', lambda file_id: zipfile.ZipFile(zpath))
    raw = tmp_path / 'bucket' / RAW_PREFIX / 'return_periods'
    fetch_mod.fetch('return_periods', dry_run=True)
    assert not raw.exists()
    fetch_mod.fetch('return_periods', only='*raster.nc*')
    assert sorted(p.relative_to(raw).as_posix() for p in raw.rglob('*') if p.is_file()) == [
        'raster.nc/ERA5/b.nc'
    ]
    assert (raw / 'raster.nc/ERA5/b.nc').read_bytes() == b'b' * 20
    capsys.readouterr()
    fetch_mod.fetch('return_periods', only='*raster.nc*')  # rerun: same size, skipped
    assert 'copied=0 skipped=1' in capsys.readouterr().out


def test_snap_ties() -> None:
    # exact halves go to the even node (np.round), the rule that matches v1
    np.testing.assert_array_equal(chaz._snap(np.array([0.5, 1.5, 2.5]), np.arange(4.0)), [0, 2, 2])
    with pytest.raises(KeyError):
        chaz._snap(np.array([3.6]), np.arange(4.0))
