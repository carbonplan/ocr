# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --tag Project=OCR

"""Compute the fire-weather filtered modal wind direction from CONUS404 data.

This script reproduces the logic from the notebook
`notebooks/fire-weather-wind-mode-reprojected.ipynb` and packages it as a
reusable CLI utility following the style of other scripts in this directory.

Steps
-----
1. Optionally start a Coiled cluster.
2. Load required CONUS404 variables (from Icechunk-backed stores via `load_conus404`).
3. Derive relative humidity and earth-relative 10 m wind components.
4. Compute sustained wind speed (& direction) and apply a gust factor.
5. Build fire weather mask using NWS-style thresholds (relative humidity, wind speed, optional temperature).
6. Classify wind directions into 8 cardinal bins for hours meeting fire weather.
7. Create a per-pixel directional histogram and take the modal direction (argmax) where any fire weather occurs.
8. Save the (native grid) modal wind direction to Zarr.
9. (Optional) Reproject to the geobox of a target dataset (default USFS wildfire risk communities) and save.

Outputs
-------
Two Zarr stores are produced (unless reprojection is skipped):
- fire_weather_wind_mode-hurs{H}_wind{W}.zarr (native CONUS404 grid)
- fire_weather_wind_mode-hurs{H}_wind{W}-reprojected.zarr (target geobox CRS)

The stored variable is `wind_direction_mode` with integer values 0-7 indexing
cardinal directions in this order: ['N','NE','E','SE','S','SW','W','NW'].

"""

from __future__ import annotations

import time

import coiled
import rich
import typer
import xarray as xr
from dask.base import optimize

from ocr.conus404 import (
    build_fire_weather_mask,
    compute_modal_wind_direction,
    compute_relative_humidity,
    compute_wind_speed_and_direction,
    load_conus404,
    reproject_mode,
    rotate_winds_to_earth,
)

console = rich.console.Console()
app = typer.Typer(help='Compute fire-weather modal wind direction (CONUS404).')


# ---------------------------------------------------------------------------
# Cluster utilities
# ---------------------------------------------------------------------------


def setup_cluster(
    name: str,
    region: str = 'us-west-2',
    min_workers: int = 2,
    max_workers: int = 50,
    worker_vm_types: str = 'm8g.xlarge',
    scheduler_vm_types: str = 'm8g.large',
) -> coiled.Cluster | None:
    """Create and return a Coiled cluster (or None if min_workers == 0).

    Returns the cluster object; a dask `Client` can be obtained via
    `cluster.get_client()`.
    """
    if min_workers <= 0:
        console.log('Skipping cluster creation (min_workers <= 0)')
        return None

    args = {
        'name': name,
        'region': region,
        'n_workers': [min_workers, max_workers],
        'tags': {'Project': 'OCR'},
        'worker_vm_types': worker_vm_types,
        'scheduler_vm_types': scheduler_vm_types,
    }
    console.log(f'Creating Coiled cluster with args: {args}')
    cluster = coiled.Cluster(**args)
    # Touch the client to ensure startup
    client = cluster.get_client()
    console.log(
        f'Cluster {cluster.name} started with {len(client.scheduler_info()["workers"])} workers (initial).'
    )
    return cluster


# ---------------------------------------------------------------------------
# Core computation steps
# ---------------------------------------------------------------------------


def save_zarr(data: xr.DataArray | xr.Dataset, path: str, overwrite: bool = True) -> None:
    mode = 'w' if overwrite else 'w-'
    console.log(f'Writing Zarr to {path} (mode={mode})')
    data.to_zarr(path, zarr_format=3, mode=mode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    hurs_threshold: float = typer.Option(15.0, help='Relative humidity threshold (%)'),
    wind_threshold: float = typer.Option(35.0, help='Wind (gust-like) threshold (mph)'),
    wind_gust_factor: float = typer.Option(
        1.4, help='Multiplier applied to sustained wind speed to approximate gusts'
    ),
    output_base: str = typer.Option(
        's3://carbonplan-ocr/input/conus404-wind-direction-modes', help='Base output path'
    ),
    target_dataset_name: str = typer.Option(
        'USFS-wildfire-risk-communities-4326', help='Catalog dataset name whose geobox is used'
    ),
    reproject: bool = typer.Option(True, help='Whether to produce a reprojected output dataset'),
    cluster_name: str = typer.Option(
        'fire-weather-distribution', help='Name for the Coiled cluster (if started)'
    ),
    min_workers: int = typer.Option(4, help='Minimum number of Coiled workers'),
    max_workers: int = typer.Option(50, help='Maximum number of Coiled workers'),
    worker_vm_types: str = typer.Option('m8g.xlarge', help='Worker VM type'),
    scheduler_vm_types: str = typer.Option('m8g.large', help='Scheduler VM type'),
    local: bool = typer.Option(
        False, help='Run locally without creating a Coiled cluster (ignores worker args)'
    ),
):
    """Compute and (optionally) reproject modal fire-weather wind direction."""
    start = time.time()
    console.rule('Fire Weather Modal Wind Direction')
    console.log(
        f'Parameters: hurs_threshold={hurs_threshold}, wind_threshold={wind_threshold}, gust_factor={wind_gust_factor}'
    )

    cluster = None
    if not local:
        cluster = setup_cluster(
            name=cluster_name,
            min_workers=min_workers,
            max_workers=max_workers,
            worker_vm_types=worker_vm_types,
            scheduler_vm_types=scheduler_vm_types,
        )
        if cluster is not None:
            cluster.get_client()  # Ensure client readiness

    console.log('Loading CONUS404 dataset')
    ds = load_conus404(add_spatial_constants=True)

    console.log('Computing relative humidity')
    hurs = compute_relative_humidity(ds)

    console.log('Rotating winds to earth-relative frame')
    earth_u, earth_v = rotate_winds_to_earth(ds)

    console.log('Computing wind speed & direction')
    wind_ds = compute_wind_speed_and_direction(earth_u, earth_v)

    console.log('Building fire weather mask')
    fire_weather_mask = build_fire_weather_mask(
        hurs,
        wind_ds,
        hurs_threshold=hurs_threshold,
        wind_threshold=wind_threshold,
        wind_gust_factor=wind_gust_factor,
    )

    console.log('Computing modal wind direction')
    mode = compute_modal_wind_direction(wind_ds['sfcWindfromdir'], fire_weather_mask)

    # Persist before writing to reduce scheduler chatter during write
    mode = optimize(mode)[0]
    mode = mode.persist()

    # Save native-grid result
    out_native = f'{output_base}/fire_weather_wind_mode-hurs{int(hurs_threshold)}_wind{int(wind_threshold)}.zarr'
    save_zarr(mode, out_native, overwrite=True)

    if reproject:
        console.log('Reprojecting output to target dataset geobox')
        result = reproject_mode(
            mode,
            ds['crs'].attrs.get('crs_wkt', 'EPSG:4326'),
            target_dataset_name=target_dataset_name,
        )
        out_reproj = f'{output_base}/fire_weather_wind_mode-hurs{int(hurs_threshold)}_wind{int(wind_threshold)}-reprojected.zarr'
        save_zarr(result, out_reproj, overwrite=True)

    if cluster is not None:
        console.log('Closing Coiled cluster')
        cluster.close()

    console.log(f'Completed in {(time.time() - start) / 60:.2f} minutes')
    console.rule('Done')


if __name__ == '__main__':
    app()
