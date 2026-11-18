from pathlib import Path

import icechunk
import numpy as np
import odc.geo.xr  # noqa: F401
import pytest
import xarray as xr

from ocr.config import StorageConfig
from ocr.risks.hurricane.etl import chaz
from ocr.risks.hurricane.etl.chaz import STORE, _detect_axis, era5_key, gcm_key, run, run_era5
from ocr.risks.hurricane.etl.fetch import EXPLORE_PREFIX, RAW_PREFIX, _clean_rel, copy_raw

STEP = 1 / 12
# Coastal points on a 1/12 deg lattice anchored half a cell off the raw minimum (the ERA5 case).
LAT = (np.arange(4) + 0.5) * STEP + 10
LON = (np.arange(5) + 0.5) * STEP - 80
PRESENT = [
    ('CESM2', 'base', 'CRH'),
    ('MIROC6', 'fut1', 'SD'),
]  # ssp245 only; every other file missing


def _points(seed: int) -> xr.Dataset:
    lat, lon = (a.ravel() for a in np.meshgrid(LAT, LON, indexing='ij'))
    keep = np.arange(lat.size) % 3 != 0  # sparse, like coastal centroids
    rng = np.random.default_rng(seed)
    bands = {b: ('points', rng.random(keep.sum())) for b in chaz.BANDS['return_periods']}
    return xr.Dataset({'lat': ('points', lat[keep]), 'lon': ('points', lon[keep]), **bands})


def _raster(seed: int) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    shape = (LAT.size, LON.size)
    bands = {b: (('lat', 'lon'), rng.random(shape)) for b in chaz.BANDS['return_periods']}
    return xr.Dataset(bands, coords={'lat': LAT, 'lon': LON})


def _era5_raster() -> xr.Dataset:
    """Raw ERA5 raster layout: bands prefixed thr_thr_*, plus per-cell point bookkeeping."""
    ds = _raster(9).assign_coords(lat=LAT - 1)
    ds = ds.rename({b: f'thr_{b}' for b in chaz.BANDS['return_periods']})
    return ds.assign(thr_points=ds['thr_thr_33'] * 0, thr_lat=ds['thr_thr_33'] * 0)


def _era5_points() -> xr.Dataset:
    return _points(8).assign(lat=lambda d: d['lat'] - 1)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StorageConfig:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    for seed, (gcm, period, variant) in enumerate(PRESENT):
        for rep, make in (('points', _points), ('raster', _raster)):
            path = tmp_path / gcm_key('return_periods', gcm, 'ssp245', period, variant, rep)
            path.parent.mkdir(parents=True, exist_ok=True)
            make(seed).to_netcdf(path, engine='h5netcdf')
    # ERA5 sorts before per-GCM/ and has its own grid; it must not set the cube's grid
    for rep, ds in (('raster', _era5_raster()), ('points', _era5_points())):
        path = tmp_path / era5_key('return_periods', rep)
        path.parent.mkdir(parents=True)
        ds.to_netcdf(path, engine='h5netcdf')
    return StorageConfig()


def _open(cfg: StorageConfig, group: str) -> xr.Dataset:
    repo = icechunk.Repository.open(cfg.icechunk_storage(f'input/{STORE}'))
    return xr.open_zarr(repo.readonly_session('main').store, group=group, consolidated=False)


def _messages(cfg: StorageConfig) -> list[str]:
    repo = icechunk.Repository.open(cfg.icechunk_storage(f'input/{STORE}'))
    return [s.message for s in repo.ancestry(branch='main')]


def test_detect_axis_off_phase() -> None:
    step, phase = _detect_axis(np.concatenate([LAT, [LAT[0] - 0.01]]))  # one off-phase point
    assert step == pytest.approx(STEP) and phase == pytest.approx(0.5)


@pytest.mark.parametrize('rep', ['points', 'raster'])
def test_roundtrip(cfg: StorageConfig, rep: str) -> None:
    run('return_periods', rep)
    ds = _open(cfg, chaz.group_name('return_periods', rep))
    assert ds['thr_33'].dims == chaz.DIMS and ds['thr_33'].dtype == np.float32
    np.testing.assert_allclose(ds['lat'], LAT) and np.testing.assert_allclose(ds['lon'], LON)
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


def test_resume(cfg: StorageConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = chaz.assemble

    def flaky(*args, **kw):
        scenario = args[3]
        calls.append(scenario)
        if scenario == 'ssp370' and calls.count('ssp370') == 1:
            raise RuntimeError('boom')
        return real(*args, **kw)

    monkeypatch.setattr(chaz, 'assemble', flaky)
    with pytest.raises(RuntimeError):
        run('return_periods')
    run('return_periods')
    assert calls == ['ssp245', 'ssp370', 'ssp370', 'ssp585']
    assert _messages(cfg)[:5] == [
        'return_periods ssp585',
        'return_periods ssp370',
        'return_periods ssp245',
        'return_periods template',
        'Repository initialized',
    ]


def test_overwrite_only_rebuilds_group(cfg: StorageConfig) -> None:
    run('return_periods', 'points')
    run('return_periods', 'raster')
    before = _open(cfg, 'return_periods_raster')['thr_33'].values
    run('return_periods', 'points', overwrite=True)
    assert _messages(cfg)[3] == 'return_periods template'  # points rebuilt from a new template
    np.testing.assert_array_equal(_open(cfg, 'return_periods_raster')['thr_33'].values, before)
    assert np.isfinite(_open(cfg, 'return_periods')['thr_33'].sel(scenario='ssp245')).any()


@pytest.mark.parametrize('rep', ['points', 'raster'])
def test_era5_roundtrip(cfg: StorageConfig, rep: str) -> None:
    run('return_periods', rep)
    run_era5('return_periods', rep)
    ds = _open(cfg, chaz.group_name('return_periods', rep, era5=True))
    assert ds['thr_33'].dims == ('lat', 'lon') and ds['thr_33'].dtype == np.float32
    assert set(ds.data_vars) == {'thr_33', 'thr_50'} and ds.attrs['forcing'] == 'ERA5'
    np.testing.assert_allclose(ds['lat'], LAT - 1)  # own grid, not the GCM one
    if rep == 'raster':
        np.testing.assert_allclose(ds['thr_33'], _raster(9)['thr_33'].values.astype('float32'))
    else:
        got, src = ds['thr_33'].values, _era5_points()
        assert np.isfinite(got).sum() == src.sizes['points']
        np.testing.assert_allclose(
            np.sort(got[np.isfinite(got)]), np.sort(src['thr_33'].values), rtol=1e-6
        )
    gcm = _open(cfg, chaz.group_name('return_periods', rep))
    np.testing.assert_allclose(gcm['lat'], LAT)
    assert _messages(cfg)[0] == f'{chaz.group_name("return_periods", rep, era5=True)} era5'


def test_era5_rerun_and_overwrite(cfg: StorageConfig) -> None:
    run('return_periods', 'points')
    before = _open(cfg, 'return_periods')['thr_33'].values
    run_era5('return_periods', 'points')
    n = len(_messages(cfg))
    run_era5('return_periods', 'points')  # already committed: no-op
    assert len(_messages(cfg)) == n
    run_era5('return_periods', 'points', overwrite=True)
    assert _messages(cfg)[:2] == ['return_periods_era5 era5'] * 2
    np.testing.assert_array_equal(_open(cfg, 'return_periods')['thr_33'].values, before)
    with pytest.raises(FileNotFoundError):
        run_era5('exceedance_intensity', 'points')


def test_raster_coords_noise_and_meridian() -> None:
    lon = np.linspace(-180, 180, 7201) + 1e-11  # float noise, 180 repeats -180
    d = xr.Dataset(coords={'lat': [0.1 + 1e-12, 0.15], 'lon': lon})
    lat, lon = chaz._raster_coords(d)
    np.testing.assert_allclose(lat, [0.1, 0.15], atol=1e-15)
    assert lon.size == 7200
    assert lon[0] == -180 and lon[-1] == pytest.approx(179.95, abs=1e-12)


def test_clean_rel() -> None:
    assert _clean_rel('return_periods/nc/ERA5/a.nc', 'return_periods') == 'nc/ERA5/a.nc'
    assert _clean_rel('/nc/a.nc', 'return_periods') == 'nc/a.nc'
    assert _clean_rel('__MACOSX/nc/._a.nc', 'return_periods') is None
    assert _clean_rel('return_periods/nc/.DS_Store', 'return_periods') is None
    assert _clean_rel('return_periods/nc/', 'return_periods') is None


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


def test_prefix(cfg: StorageConfig, tmp_path: Path) -> None:
    run('return_periods', prefix='scratch/me')
    assert (tmp_path / 'scratch/me' / STORE).is_dir()
    assert not (tmp_path / 'input' / STORE).exists()
