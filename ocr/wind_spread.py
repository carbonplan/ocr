import rioxarray

from ocr import catalog
from ocr.console import console
from ocr.coordinates import convert_coords, lon_to_180
from ocr.subsetting import subset_region_xy
from ocr.utils import interpolate_to_30
from ocr.wind import (
    apply_mode_calc,
    apply_wind_directional_convolution,
    classify_wind_directions,
    create_composite_bp_map,
    create_finescale_wind_direction,
)


def main(bounding_box, buffer: int = 2000):
    console.print(f'Processing fire spread with bounding box: {bounding_box}')
    x_min, x_max = bounding_box[0] - buffer, bounding_box[2] + buffer
    y_min, y_max = bounding_box[1] - buffer, bounding_box[3] + buffer
    console.print('Loading climate datasets... ')
    riley = {
        '2011': catalog.get_dataset('2011-climate-run').to_xarray(
            is_icechunk=True, xarray_open_kwargs={'chunks': {}}
        ),
        '2047': catalog.get_dataset('2047-climate-run').to_xarray(
            is_icechunk=True, xarray_open_kwargs={'chunks': {}}
        ),
    }

    console.print('Subsetting the region of interest...')
    subset = subset_region_xy(riley['2011'], [x_min, x_max], [y_min, y_max])

    console.print('Converting coordinates...')
    (lon_min, lat_max), (lon_max, lat_min) = convert_coords(
        [(x_min, y_max), (x_max, y_min)], from_crs='EPSG:5070', to_crs='EPSG:4326'
    )

    lat_max += 0.2
    lat_min -= 0.2
    lon_max += 0.15
    lon_min -= 0.15

    console.print('Loading 30m product as a template...')
    rps_30 = rioxarray.open_rasterio(
        's3://carbonplan-risks/wildfirecommunities/RPS_CA.tif', chunks={}
    )
    rps_30 = subset_region_xy(rps_30, [x_min, x_max], [y_min, y_max]).sel(band=1)

    console.print('Interpolating to 30m...')
    subset_30 = interpolate_to_30(subset, rps_30)
    subset_30 = subset_30.rio.write_crs('EPSG:5070')

    console.print('Applying wind directional convolution...')
    blurred_bp = apply_wind_directional_convolution(subset_30['BP'], iterations=3)

    console.print('Loading wind dataset for fire weather days...')
    important_days = catalog.get_dataset('era5-fire-weather-days').to_xarray(
        is_icechunk=False, xarray_open_kwargs={'engine': 'zarr'}
    )
    important_days = lon_to_180(important_days)

    console.print('Subsetting wind data...')
    subset_wind = important_days.sel(
        latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
    )
    wind_directions = subset_wind['sfcWindfromdir']

    console.print('Classifying wind directions...')
    direction_indices = classify_wind_directions(wind_directions).chunk(dict(time=-1))
    direction_modes = apply_mode_calc(direction_indices).compute()

    console.print('Creating finescale wind direction...')
    direction_modes_reprojected = create_finescale_wind_direction(blurred_bp, direction_modes)

    console.print('Creating composite BP map...')
    wind_informed_bp = create_composite_bp_map(blurred_bp, direction_modes_reprojected)

    console.print('Loading CRPS for registration...')
    crps = rioxarray.open_rasterio(
        's3://carbonplan-risks/wildfirecommunities/CRPS_CA.tif', chunks={}
    )
    subset_crps = crps.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))

    console.print('Reprojecting wind informed BP...')
    wind_informed_bp = wind_informed_bp.rio.reproject_match(subset_crps)

    console.print('Trimming wind informed BP...')
    wind_informed_bp = wind_informed_bp.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))

    console.print('Saving wind informed BP...')
    wind_informed_bp.to_dataset(name='BP').to_zarr(
        's3://carbonplan-risks/ocr/v0/intermediates/bp.zarr', mode='w'
    )

    console.print('Process completed successfully.')
