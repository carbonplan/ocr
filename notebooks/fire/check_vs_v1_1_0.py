"""Recompute fire regions on Coiled with the shipped score config and diff them against v1.1.0.

Rerun after bumping proj, proj-data or rasterio (check_proj_grids pins them). One map task per
region on score's VMs (cli.DEFAULT conda extras, no overrides); each puts a JSON with its PROJ
environment and per-variable pixel diffs on S3, then this script prints a table.

    pixi run -e dev python notebooks/fire/check_vs_v1_1_0.py [region_id ...]
"""

import json
import sys

import coiled
import obstore

from ocr.cli import job_config, job_finished, wait
from ocr.config import StorageConfig

# Where NOAA HARN grids overlap, so the most PROJ-sensitive regions
REGIONS = sys.argv[1:] or [
    'y4_x31',
    'y4_x32',
    'y4_x33',
    'y5_x24',
    'y5_x25',
    'y5_x26',
    'y5_x27',
    'y5_x34',
]
KEY = 'scratch/nrh/fire/check-vs-v1.1.0'

VM_CODE = """
import json, os, sys
from importlib.metadata import distributions
from pathlib import Path

import icechunk, numpy as np, obstore, pyproj, rasterio, xarray as xr

from ocr.config import StorageConfig
from ocr.risks.fire.grid import ChunkingConfig
from ocr.risks.fire.model import calculate_wind_adjusted_risk

region, key = os.environ['COILED_BATCH_TASK_INPUT'], os.environ['CHECK_KEY']
store = StorageConfig().object_store()
share = Path(sys.prefix) / 'share' / 'proj'
conda = {m['name']: f"{m['version']} {m['build']}" for m in (json.loads(f.read_text()) for f in (Path(sys.prefix) / 'conda-meta').glob('*.json'))}
info = {
    'region': region,
    'proj': rasterio.__proj_version__,
    'gdal': rasterio.__gdal_version__,
    'proj_data_pkg': conda.get('proj-data'),
    'PROJ_NETWORK': os.environ.get('PROJ_NETWORK'),
    'PROJ_DATA': os.environ.get('PROJ_DATA'),
    'local_grids': len(list(share.glob('*.tif'))),
    # odc-geo projects extents with pyproj, a PyPI wheel with its own PROJ and data dir (no grids)
    'pyproj': f'{pyproj.__version__} (PROJ {pyproj.proj_version_str})',
    'pyproj_network': pyproj.network.is_network_enabled(),
    'conda': conda,
    'pip': {d.metadata['Name'].lower(): d.version for d in distributions()},
}
# Before compute, so a crashed task still leaves its environment
obstore.put(store, f'{key}/{region}.json', json.dumps(info).encode())
y, x = ChunkingConfig().region_id_to_latlon_slices(region_id=region)
ds = calculate_wind_adjusted_risk(y_slice=y, x_slice=x).compute()  # runs check_proj_grids
storage = icechunk.s3_storage(bucket='carbonplan-ocr', prefix='output/fire-risk/tensor/production/v1.1.0/ocr.icechunk', region='us-west-2', from_env=True)
prod = xr.open_zarr(icechunk.Repository.open(storage).readonly_session('main').store, consolidated=False).sel(latitude=y, longitude=x)


def diff_px(a, b):
    if a.shape != b.shape:
        return f'shape {a.shape} vs {b.shape}'
    return int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())


info['px_vs_prod'] = {v: diff_px(ds[v].values, prod[v].values) for v in ds.data_vars if v in prod}
print(json.dumps({k: v for k, v in info.items() if k not in ('conda', 'pip')}, indent=1))
obstore.put(store, f'{key}/{region}.json', json.dumps(info).encode())
"""
SCRIPT = f"#!/bin/bash\nset -e\npython - <<'EOF'\n{VM_CODE}\nEOF\n"

job = job_config('score', {}, None)
job['env'] |= {'CHECK_KEY': KEY}
j = coiled.batch.run(command=SCRIPT, name='fire-check-vs-v1-1-0', map_over_values=REGIONS, **job)
print(f'logs: coiled batch logs {j["cluster_id"]}')
wait(j, job_finished)

store = StorageConfig().object_store()
results = {r: json.loads(bytes(obstore.get(store, f'{KEY}/{r}.json').bytes())) for r in REGIONS}
env = {
    k: {str(i[k]) for i in results.values()}
    for k in (
        'proj',
        'gdal',
        'proj_data_pkg',
        'PROJ_NETWORK',
        'PROJ_DATA',
        'local_grids',
        'pyproj',
        'pyproj_network',
    )
}
for k, v in env.items():
    print(f'{k:16} {", ".join(sorted(v))}')
for r, i in results.items():
    px = i.get('px_vs_prod')
    print(
        f'{r:8} '
        + (
            'CRASHED (see logs)'
            if px is None
            else f'{sum(px.values()) if all(isinstance(n, int) for n in px.values()) else px} px differ'
        )
    )
ok = all(i.get('px_vs_prod') and not any(i['px_vs_prod'].values()) for i in results.values())
print('ALL REGIONS == v1.1.0' if ok else 'DIFFERENCES OR FAILURES')
