"""Fire score on a fake 2x2 grid of RxR-pixel regions, on local storage (no catalog, no S3)."""

import json
import types
from pathlib import Path

import icechunk
import numpy as np
import odc.geo.xr  # noqa: F401 - registers .odc accessor
import pytest
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk
from typer.testing import CliRunner

from ocr import cli, stages
from ocr.config import OutputPaths, Risk, StorageConfig
from ocr.risks.fire import model, score

R = 512  # big enough that topozarr's chunks (256) sit inside a region shard
LAT, LON = 0.5 + np.arange(2.0 * R), -100.5 + np.arange(2.0 * R)
IDS = ['y0_x0', 'y0_x1', 'y1_x0', 'y1_x1']
OUT = OutputPaths(Risk.FIRE, 'v0.0.1')
CHECK_PROJ_GRIDS = model.check_proj_grids  # the fixture no-ops it


def _slices(region_id: str) -> tuple[slice, slice]:
    y, x = int(region_id[1]), int(region_id[-1])
    return slice(LAT[R * y], LAT[R * y + R - 1]), slice(LON[R * x], LON[R * x + R - 1])


def _risk(*, y_slice: slice, x_slice: slice) -> xr.Dataset:
    # Distinct float64 values per cell and variable, so a misplaced or recast write shows.
    coords = xr.Dataset(coords={'latitude': LAT, 'longitude': LON}).sel(
        latitude=y_slice, longitude=x_slice
    )
    seed = int(coords.latitude[0] * 10 + coords.longitude[0] + 101)
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {v: (('latitude', 'longitude'), rng.random((R, R))) for v in model.VARIABLE_ATTRS},
        coords=coords.coords,
    )


@pytest.fixture(autouse=True)
def grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    monkeypatch.delenv(stages.TASK_INPUT, raising=False)
    g = types.SimpleNamespace(
        ds=xr.Dataset(coords={'latitude': LAT, 'longitude': LON, 'spatial_ref': 0}),
        chunks={'latitude': R, 'longitude': R},
        valid_region_ids=IDS,
        region_id_to_latlon_slices=lambda region_id: _slices(region_id),
    )
    monkeypatch.setattr(score, '_grid', lambda: g)
    monkeypatch.setattr(model, 'calculate_wind_adjusted_risk', _risk)
    monkeypatch.setattr(model, 'check_proj_grids', lambda: None)  # offline; tested below


def _repo() -> icechunk.Repository:
    return icechunk.Repository.open(StorageConfig().icechunk_storage(OUT.store))


def _open() -> xr.Dataset:
    session = _repo().readonly_session('main')
    return xr.open_zarr(session.store, group='0', consolidated=False).load()


@pytest.mark.parametrize(
    ('lon', 'ok'), [(model.PROJ_PROBE_LON, True), (-107.92266736878591, False)]
)
def test_check_proj_grids(lon: float, ok: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    # Grid transform passes; the null transform (grids missing) raises.
    import rasterio.warp

    monkeypatch.setattr(rasterio.warp, 'transform', lambda *_: ([lon], [40.448]))
    if ok:
        CHECK_PROJ_GRIDS()
    else:
        with pytest.raises(RuntimeError, match='datum grid'):
            CHECK_PROJ_GRIDS()


def test_check_proj_grids_rejects_other_proj(monkeypatch: pytest.MonkeyPatch) -> None:
    import rasterio

    monkeypatch.setattr(rasterio, '__proj_version__', '9.9.0')
    with pytest.raises(RuntimeError, match='want 9.8.1'):
        CHECK_PROJ_GRIDS()


def test_check_proj_grids_rejects_other_proj_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model, 'PROJ_DATA_VERSION', '0.0')
    with pytest.raises(RuntimeError, match='proj-data 0.0 not installed'):
        CHECK_PROJ_GRIDS()


def test_store_values_are_the_computation() -> None:
    # Values must match the old pipeline: the computation, cast to the float32 template.
    for r in score.parts(OUT):
        score.write_region(OUT, r)
    ds = _open()
    for r in IDS:
        y, x = _slices(r)
        got = ds.sel(latitude=y, longitude=x)
        want = _risk(y_slice=y, x_slice=x).astype('float32')
        for v in model.VARIABLE_ATTRS:
            np.testing.assert_array_equal(got[v].values, want[v].values)
            assert got[v].dtype == np.float32
    assert {v: dict(ds[v].attrs) for v in model.VARIABLE_ATTRS} == model.VARIABLE_ATTRS
    assert ds.attrs['version'] == 'v0.0.1'
    assert ds.odc.crs.epsg == 4326
    # topozarr's chunks inside one shard per region, so writers never share an object.
    assert ds.rps_2011.encoding['chunks'] == (256, 256)
    assert ds.rps_2011.encoding['shards'] == (R, R)
    assert 'zarr_conventions' in ds.attrs


def test_resume_plans_only_missing_regions() -> None:
    assert score.parts(OUT) == IDS
    for r in IDS[:3]:
        score.write_region(OUT, r)
    assert score.parts(OUT) == ['y1_x1']
    y, x = _slices('y1_x1')
    assert np.isnan(_open().rps_2011.sel(latitude=y, longitude=x)).all()
    assert score.parts(OUT, region_ids=['y0_x0', 'y1_x1']) == ['y1_x1']
    region_ids = {s.metadata.get('region_id') for s in _repo().ancestry(branch='main')}
    assert region_ids == {None, *IDS[:3]}


@pytest.mark.parametrize('region_ids', [['nope'], []])
def test_bad_region_ids_raise(region_ids: list[str]) -> None:
    with pytest.raises(ValueError):
        score.parts(OUT, region_ids=region_ids)


def test_same_base_sessions_all_commit() -> None:
    # Every task opens its session before any other commits; same-row regions share latitude.
    score.parts(OUT)
    repo = _repo()
    sessions = [repo.writable_session('main') for _ in IDS]
    for s, r in zip(sessions, IDS, strict=True):
        y, x = _slices(r)
        to_icechunk(_risk(y_slice=y, x_slice=x), s, group='0', region='auto')
    for s, r in zip(sessions, IDS, strict=True):
        s.commit(r, metadata={'region_id': r}, rebase_with=icechunk.ConflictDetector())
    assert score.parts(OUT) == []
    assert not np.isnan(_open().rps_2011).any()


def test_stage_overwrite_rewrites_every_region() -> None:
    spec = stages.RISKS[Risk.FIRE]
    stages.score(spec, OUT, overwrite=False)
    stages.score(spec, OUT, overwrite=False)  # resume: nothing left to write
    stages.score(spec, OUT, overwrite=True)
    commits = [s.metadata.get('region_id') for s in _repo().ancestry(branch='main')]
    assert sorted(c for c in commits if c) == IDS


def test_new_regions_drop_stale_pyramid() -> None:
    # Group "0" changing under a pyramid drops its levels, as flood-etl --overwrite does.
    spec = stages.RISKS[Risk.FIRE]
    kw = {'x_dim': 'longitude', 'y_dim': 'latitude', 'levels': 2}
    stages.score(spec, OUT, overwrite=False, region_ids=IDS[:2])
    stages.pyramid(spec, OUT, overwrite=False, **kw)
    stages.score(spec, OUT, overwrite=False, region_ids=IDS[:2])  # nothing new: levels kept
    assert '1' in zarr.open_group(_repo().readonly_session('main').store, mode='r')
    stages.score(spec, OUT, overwrite=False)
    root = zarr.open_group(_repo().readonly_session('main').store, mode='r')
    assert list(root.group_keys()) == ['0'] and 'multiscales' not in root.attrs
    assert score.parts(OUT) == []


def test_score_then_pyramid_through_cli() -> None:
    kwargs = {'pyramid': {'x_dim': 'longitude', 'y_dim': 'latitude', 'levels': 2}}
    args = ['stage', '--version', 'v0.0.1', '--kwargs', json.dumps(kwargs), 'fire']
    r = CliRunner().invoke(cli.app, [*args, 'score', 'pyramid'])
    assert r.exit_code == 0, r.output
    session = _repo().readonly_session('main')
    level = xr.open_zarr(session.store, group='1', consolidated=False)
    assert level.sizes == {'latitude': R, 'longitude': R}


def test_real_grid_encoding_and_edge_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real CONUS 30 m shape and 6000x4500 regions (fake values). The template is metadata only; the
    # two written regions are the small last row's corner and its neighbour (partial edge shards).
    shape, size = (97579, 208881), {'latitude': 6000, 'longitude': 4500}
    lat, lon = np.arange(float(shape[0])), np.arange(float(shape[1]))
    ids = ['y16_x45', 'y16_x46']  # 17 rows x 47 cols; the last row is 1579 tall, last col 1881 wide

    def slices(region_id: str) -> tuple[slice, slice]:
        y, x = (int(p[1:]) for p in region_id.split('_'))
        ys, xs = y * size['latitude'], x * size['longitude']
        y_end = min(ys + size['latitude'], shape[0]) - 1
        x_end = min(xs + size['longitude'], shape[1]) - 1
        return slice(lat[ys], lat[y_end]), slice(lon[xs], lon[x_end])

    def risk(*, y_slice: slice, x_slice: slice) -> xr.Dataset:
        c = xr.Dataset(coords={'latitude': lat, 'longitude': lon}).sel(
            latitude=y_slice, longitude=x_slice
        )
        values = np.full((c.sizes['latitude'], c.sizes['longitude']), int(x_slice.start), 'float32')
        return xr.Dataset({'rps_2011': (('latitude', 'longitude'), values)}, coords=c.coords)

    g = types.SimpleNamespace(
        ds=xr.Dataset(coords={'latitude': lat, 'longitude': lon, 'spatial_ref': 0}),
        chunks=size,
        valid_region_ids=ids,
        region_id_to_latlon_slices=slices,
    )
    monkeypatch.setattr(score, '_grid', lambda: g)
    monkeypatch.setattr(model, 'calculate_wind_adjusted_risk', risk)

    for r in score.parts(OUT):
        score.write_region(OUT, r)
    assert score.parts(OUT) == []
    ds = xr.open_zarr(_repo().readonly_session('main').store, group='0', consolidated=False)
    assert ds.rps_2011.encoding['chunks'] == (375, 375)
    assert ds.rps_2011.encoding['shards'] == (6000, 4500)
    for r in ids:
        y, x = slices(r)
        got = ds.rps_2011.sel(latitude=y, longitude=x).values
        assert got.shape == (1579, 4500 if r == 'y16_x45' else 1881)
        assert (got == x.start).all()
    assert np.isnan(ds.rps_2011.isel(latitude=95999, longitude=0))  # row y15: unwritten
