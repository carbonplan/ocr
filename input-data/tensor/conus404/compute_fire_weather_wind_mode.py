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
from typing import cast

import coiled
import rich
import typer
import xarray as xr
import xclim
from dask.base import optimize
from odc.geo import CRS
from odc.geo.xr import assign_crs

from ocr.datasets import catalog, load_conus404
from ocr.risks.fire import classify_wind_directions, direction_histogram, nws_fire_weather

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
    allow_fallback: bool = True,
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
        'allow_fallback': allow_fallback,
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


def compute_relative_humidity(ds: xr.Dataset) -> xr.DataArray:
    """Compute relative humidity from temperature and dewpoint in ds."""
    with xr.set_options(keep_attrs=True):
        hurs = xclim.indicators.atmos.relative_humidity_from_dewpoint(tas=ds['T2'], tdps=ds['TD2'])
    hurs.name = 'hurs'
    return hurs


def rotate_winds_to_earth(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Rotate grid-relative 10 m winds (U10,V10) to earth-relative components.

    Uses SINALPHA / COSALPHA convention from WRF (same as notebook).
    """
    with xr.set_options(keep_attrs=True):
        earth_u = ds.U10 * ds.COSALPHA - ds.V10 * ds.SINALPHA
        earth_v = ds.V10 * ds.COSALPHA + ds.U10 * ds.SINALPHA
    earth_u.name = 'u10_earth'
    earth_v.name = 'v10_earth'
    return earth_u, earth_v


def compute_wind_speed_and_direction(u10: xr.DataArray, v10: xr.DataArray) -> xr.Dataset:
    """Derive hourly wind speed (m/s) and direction (degrees from) using xclim."""
    winds = xclim.indicators.atmos.wind_speed_from_vector(uas=u10, vas=v10)
    # xclim returns a tuple-like (speed, direction). Merge keeps names (sfcWind, sfcWindfromdir)
    wind_ds = xr.merge(winds)
    return wind_ds


def build_fire_weather_mask(
    hurs: xr.DataArray,
    wind_ds: xr.Dataset,
    *,
    hurs_threshold: float,
    wind_threshold: float,
    wind_gust_factor: float = 1.4,
) -> xr.DataArray:
    """Compute a boolean fire weather mask.

    Applies gust factor to sustained wind speed before thresholding.
    """
    # Convert sustained to approximate gusts via multiplier
    # reason that wind gusts are typically ~40% higher than average wind speed
    # and we want to base this on wind gusts (need a citation for this)
    gust_like = wind_ds['sfcWind'] * wind_gust_factor
    mask = nws_fire_weather(
        hurs, hurs_threshold, gust_like, wind_threshold, tas=None, tas_threshold=None
    )
    mask.name = 'fire_weather_mask'
    return mask


def compute_modal_wind_direction(
    direction: xr.DataArray,
    fire_weather_mask: xr.DataArray,
) -> xr.DataArray:
    """Compute modal wind direction (0-7) for hours satisfying fire weather.

    Direction codes follow: 0=N,1=NE,2=E,3=SE,4=S,5=SW,6=W,7=NW
    """
    direction_indices = classify_wind_directions(direction)
    masked = direction_indices.where(fire_weather_mask)
    fraction = direction_histogram(masked)
    # Help static type checkers – ensure fraction is treated as DataArray
    fraction = cast(xr.DataArray, fraction)
    assert isinstance(fraction, xr.DataArray)

    # Identify pixels with any fire-weather hours (probabilities sum to 1 else 0)
    any_fire_weather = fraction.sum(dim='wind_direction') > 0

    mode = fraction.argmax(dim='wind_direction').where(any_fire_weather)  # type: ignore[attr-defined]
    # Optimize graph early
    mode = optimize(mode)[0]
    mode.name = 'wind_direction_mode'
    mode.attrs.update(
        {
            'long_name': 'Modal wind direction during fire-weather hours',
            'description': 'Most frequent of 8 cardinal directions during hours meeting fire weather criteria',
            'direction_labels': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
            'fire_weather_definition': 'RH < hurs_threshold (%), gust_like_wind > wind_threshold (mph)',
        }
    )
    return mode


def save_zarr(data: xr.DataArray | xr.Dataset, path: str, overwrite: bool = True) -> None:
    mode = 'w' if overwrite else 'w-'
    console.log(f'Writing Zarr to {path} (mode={mode})')
    data.to_zarr(path, zarr_format=2, consolidated=True, mode=mode)


def reproject_mode(
    mode: xr.DataArray,
    src_crs_wkt: str,
    target_dataset_name: str,
    *,
    chunk_lat: int = 6000,
    chunk_lon: int = 4500,
) -> xr.DataArray:
    """Reproject the modal wind direction to the geobox of a target dataset."""
    tgt = catalog.get_dataset(target_dataset_name).to_xarray().astype('float32')
    tgt = assign_crs(tgt, crs='EPSG:4326')
    geobox = tgt.odc.geobox

    src_crs = CRS(src_crs_wkt)
    mode_src = assign_crs(mode, crs=src_crs)

    console.log('Reprojecting modal wind direction to target geobox')
    result = (
        mode_src.odc.reproject(geobox, resampling='nearest')
        .astype('float32')
        .chunk({'latitude': chunk_lat, 'longitude': chunk_lon})
    )
    result = optimize(result)[0]
    result.attrs.update({'reprojected_to': target_dataset_name})
    return result


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
        's3://carbonplan-ocr/input-data/conus404-wind-direction-modes', help='Base output path'
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
