import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from geoparquet_lod import validate

from ocr.lod import COMPLETE_ZOOM, EVERY_FROM_ZOOM, POLYGONS_FROM_ZOOM, write_buildings_lod
from ocr.stages import RISKS

GEO = {
    'version': '1.1.0',
    'primary_column': 'geometry',
    'columns': {'geometry': {'encoding': 'WKB', 'geometry_types': ['Polygon']}},
}


@pytest.fixture
def buildings(tmp_path):
    rng = np.random.default_rng(3)
    n = 3000
    x = -105.4 + rng.random(n) * 0.3
    y = 39.9 + rng.random(n) * 0.2
    boxes = shapely.box(x, y, x + 1e-4, y + 1e-4)
    risk = rng.random(n).astype(np.float32)
    risk[: n // 3] = 0
    t = pa.table(
        {
            'GEOID': pa.array([f'08013{i:010d}' for i in range(n)]),
            'rps_2011': pa.array(risk),
            'rps_2047': pa.array(risk * 1.5),
            'bp_2011': pa.array(rng.random(n).astype(np.float32)),
            'geometry': pa.array(shapely.to_wkb(boxes), pa.binary()),
        }
    ).replace_schema_metadata({'geo': json.dumps(GEO), 'overture_release': 'r1'})
    path = tmp_path / 'buildings.parquet'
    pq.write_table(t, path)
    return path


def test_buildings_lod_keeps_every_building(buildings, tmp_path, monkeypatch):
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path / 'store'))
    fire = RISKS['fire']
    write_buildings_lod(
        buildings,
        'output/fire/buildings.parquet',
        shown=fire.lod_shown,
        carry=fire.lod_carry,
        workers=1,
        log=lambda *_: None,
    )
    dest = tmp_path / 'store/output/fire/buildings.parquet'
    validate(str(dest))

    out = pq.ParquetFile(dest)
    src = pq.read_table(buildings)
    assert out.metadata.num_rows == src.num_rows
    assert set(src.column_names) <= set(out.schema_arrow.names)
    assert out.schema_arrow.metadata[b'overture_release'] == b'r1'  # source metadata kept

    lod = json.loads(out.schema_arrow.metadata[b'geo'])['lod_columns']
    assert max(lod['zooms']) == COMPLETE_ZOOM
    # tiles carry half precision; the source columns keep theirs
    assert lod['attributes'] == [
        {'name': 'rps_2011', 'type': 'halffloat'},
        {'name': 'rps_2047', 'type': 'halffloat'},
    ]
    assert out.schema_arrow.field('rps_2011').type == src.schema.field('rps_2011').type
    t = pq.read_table(dest, columns=['rps_2011', 'rps_2047', 'lod'])
    lods = t['lod'].combine_chunks()
    risky = (t['rps_2011'].to_numpy() > 0) & (t['rps_2047'].to_numpy() > 0)

    def drawn(zoom):
        name = next(n for n, c in lod['lods'].items() if zoom in c['zooms'])
        return lods.field(name).is_valid().to_numpy(zero_copy_only=False)

    # the coarsest zooms draw only shown buildings; zero-risk ones join from POLYGONS_FROM_ZOOM
    assert drawn(0).any()
    assert not any((drawn(z) & ~risky).any() for z in range(POLYGONS_FROM_ZOOM))
    assert any((drawn(z) & ~risky).any() for z in range(POLYGONS_FROM_ZOOM, EVERY_FROM_ZOOM))
    assert drawn(EVERY_FROM_ZOOM).all()


def test_buildings_lod_reruns_on_its_own_output(buildings, tmp_path, monkeypatch):
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path / 'store'))
    fire = RISKS['fire']
    kw = {'shown': fire.lod_shown, 'carry': fire.lod_carry, 'workers': 1, 'log': lambda *_: None}
    write_buildings_lod(buildings, 'a/buildings.parquet', **kw)
    first = tmp_path / 'store/a/buildings.parquet'
    write_buildings_lod(first, 'b/buildings.parquet', **kw)
    second = tmp_path / 'store/b/buildings.parquet'
    validate(str(second))
    a, b = pq.ParquetFile(first), pq.ParquetFile(second)
    assert b.metadata.num_rows == a.metadata.num_rows
    assert b.schema_arrow.names == a.schema_arrow.names
    assert b.schema_arrow.metadata[b'overture_release'] == b'r1'
