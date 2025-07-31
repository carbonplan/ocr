import typing

import numpy as np
import xarray as xr

from ocr.console import console


# Depreciate? - potentially unused
def generate_angles() -> dict[str, float]:
    """Generate a dictionary mapping cardinal/ordinal directions to their starting angles in degrees.

    Returns
    -------
    dict[str, float]
        A dictionary mapping cardinal/ordinal directions to their starting angles in degrees.
    """
    angles_dict = {}
    angles = np.arange(22.5, 360, 45).astype('float32')
    wind_direction_labels = ['NE', 'N', 'NW', 'W', 'SW', 'S', 'SE', 'E']
    for direction, angle in zip(wind_direction_labels, angles):
        angles_dict[direction] = angle
    return angles_dict


def generate_weights(
    method: typing.Literal['skewed', 'circular_focal_mean'] = 'skewed',
    kernel_size: float = 81.0,
    circle_diameter: float = 35.0,
) -> np.ndarray:
    """Generate a 2D array of weights for a circular kernel."""
    if method == 'skewed':
        x, y = np.meshgrid(
            np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1),
            np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1),
        )
        distances = np.sqrt(x**2 + y**2)
        inside = distances <= circle_diameter // 2 + 1
        weights = inside / inside.sum()

    elif method == 'circular_focal_mean':
        x, y = np.meshgrid(
            np.arange(-(kernel_size // 2), kernel_size // 2 + 1),
            np.arange(-(kernel_size // 2), kernel_size // 2 + 1),
        )
        distortion_x = 1.5
        distortion_y = 1 / distortion_x
        distances = np.sqrt((x / distortion_x) ** 2 + (y / distortion_y) ** 2)
        inside = distances <= circle_diameter // 2
        weights = inside / inside.sum()
        weights = np.roll(weights, -14)

    else:
        raise ValueError(f'Unknown method: {method}')

    return weights


def generate_wind_directional_kernels(
    kernel_size: float = 81.0, circle_diameter: float = 35.0
) -> dict[str, np.ndarray]:
    from scipy.ndimage import rotate

    """Generate a dictionary of 2D arrays of weights for circular kernels oriented in different directions.
    Parameters
    ----------
    kernel_size : float, optional
        The size of the kernel, by default 81.0
    circle_diameter : float, optional
        The diameter of the circle, by default 35.0
    Returns
    -------
    dict[str, np.ndarray]
        A dictionary of 2D arrays of weights for circular kernels oriented in different directions.
    """
    weights_dict = {}
    rotating_angles = np.arange(0, 360, 45)
    wind_direction_labels = ['W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW']
    for angle, direction in zip(rotating_angles, wind_direction_labels):
        weights_dict[direction] = rotate(
            generate_weights(
                method='skewed', kernel_size=kernel_size, circle_diameter=circle_diameter
            ),
            angle=angle,
        )
        if angle in [45, 135, 225, 315]:
            weights_dict[direction] = weights_dict[direction][
                17:98, 17:98
            ]  # TODO, @orianac, i presume this cropping only applies to kernel_size=81.0, circle_diameter=35.0. If so, what should the cropping be for other kernel sizes and circle diameters?
    weights_dict['circular'] = generate_weights(
        method='circular_focal_mean', kernel_size=kernel_size, circle_diameter=circle_diameter
    )

    # re-normalize all weights to ensure sum equals 1.0
    for direction in weights_dict:
        weights_dict[direction] = weights_dict[direction] / weights_dict[direction].sum()
    return weights_dict


def apply_wind_directional_convolution(
    da: xr.DataArray, iterations: int = 3, kernel_size: float = 81.0, circle_diameter: float = 35.0
) -> xr.DataArray:
    """Apply a directional convolution to a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to apply the convolution to.
    iterations : int, optional
        The number of iterations to apply the convolution, by default 3
    kernel_size : float, optional
        The size of the kernel, by default 81.0
    circle_diameter : float, optional
        The diameter of the circle, by default 35.0

    Returns
    -------
    xr.DataArray
        The DataArray with the directional convolution applied
    """
    import cv2 as cv

    # TODO: must scale the size of the kernel according to the latitude. Can either
    # be done before entering this function to calculate the kernel_size
    # argument or inside this function and pass latitude into the convolution
    # instead and calculate the kernel size here.
    weights_dict = generate_wind_directional_kernels(
        kernel_size=kernel_size, circle_diameter=circle_diameter
    )
    # do the spreading in each of the 8 directions with the correct weights
    # TODO: @orianac, do we want to support dask arrays here?
    # initialize dataset with the original burn probability
    spread_results = xr.Dataset(
        data_vars={var_name: (da.dims, da.values) for var_name in weights_dict.keys()},
        coords=da.coords,
    )
    for direction, weights in weights_dict.items():
        # spread_results[direction] = xr.zeros_like(da)
        for i in np.arange(
            iterations
        ):  # TODO, @orianac, is there a reason we are iterating over (iterations) without using the index. It appears that the output is the same regardless of the number of iterations.
            spread_results[direction] = (
                da.dims,
                cv.filter2D(spread_results[direction].values, -1, weights),
            )
    return spread_results


def classify_wind_directions(wind_direction_ds: xr.Dataset) -> xr.Dataset:
    # todo - make tests that ensure that our orientation is always initialized
    # to the north (0 index = north like direction_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'circular']
    #  and that it's centered (i.e. North is -22.5 to 22.5)
    rotating_angles = np.arange(22.5, 360, 45)
    wind_direction_ds %= 337.5  # this handles the North winds that are just under 360 degrees
    classified = xr.apply_ufunc(
        lambda wind_direction_ds: np.where(
            np.isnan(wind_direction_ds),
            -1,  # Assign a placeholder value (e.g., -1) for NaNs
            # TODO let's remove this placeholder because we want to know
            # if there are NaNs and don't want to unwittingly mask them
            np.digitize(wind_direction_ds, bins=rotating_angles),
        ),
        wind_direction_ds,
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int],
    )
    # Todo: make a test to ensure that this always produces integers between 0 and 8 inclusive
    return classified


def compute_mode(arr):
    """Compute the mode of an array, ignoring placeholder values (-1)."""
    arr = arr[arr != -1]  # Exclude placeholder values
    if len(arr) == 0:  # If all values are NaN, return a default or NaN. TODO: remove this
        # eventually because we want to know about nans
        return -1
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]


def apply_mode_calc(direction_indices_ds: xr.Dataset) -> xr.Dataset:
    return xr.apply_ufunc(
        compute_mode,
        direction_indices_ds,
        input_core_dims=[['time']],  # Apply along the 'time' dimension
        output_core_dims=[[]],  # Result is scalar per pixel
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int],
    )


# Depreciate? - potentially unused
def create_finescale_wind_direction(bp: xr.Dataset, wind_direction: xr.Dataset) -> xr.Dataset:
    from rasterio.warp import Resampling

    wind_direction = wind_direction.rio.write_crs('EPSG:4326')
    # bp = bp.rio.write_crs('EPSG:5070')
    # doing nearest neighbor resampling here introduces strong artifacts along gridcell boundaries.
    # TODO: consider ways of creating a smooth transition between gridcells, options include:
    # - instead of using the mode direction, do a weighted average of the different winds and then
    # interpolate between those to create a smooth surface of weights
    # - do a smoothing step afterwards (not preferred)
    # - leave as-is with explanation (not preferred)
    # - something else?
    # also: reevaluate preformance for CONUS404 dataset
    wind_direction_reprojected = wind_direction.rio.reproject_match(
        bp, resampling=Resampling.nearest
    )
    # if negative (likely b/c it's nan) then cast it to -1 for now- TODO: we actually want to raise an error!
    wind_direction_reprojected = wind_direction_reprojected.where(
        wind_direction_reprojected >= 0, -1
    )
    return wind_direction_reprojected


def create_composite_bp_map(bp: xr.Dataset, wind_directions) -> xr.Dataset:
    direction_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'circular']
    # reorder the differently blurred BP and then turn it into a DataArray and assign the coords
    bp_da = bp[direction_labels].to_array(dim='direction').assign_coords(direction=direction_labels)
    # select the entry that corresponds to the wind direction index
    # TODO: let's test this to confirm it's working as expected
    return bp_da.isel(direction=wind_directions)


def classify_wind(
    climate_run_subset: xr.Dataset, direction_modes_sfc: xr.Dataset, rps_30_subset: xr.Dataset
) -> xr.Dataset:
    from odc.geo.xr import assign_crs

    from ocr.risks.fire import apply_wind_directional_convolution, create_composite_bp_map

    # Build and apply wind adjustment
    blurred_bp = apply_wind_directional_convolution(climate_run_subset['BP'], iterations=3)

    blurred_bp = assign_crs(blurred_bp, crs='EPSG:4326')
    # Switched to xarray interp_like to since both datasets have matching EPSG codes.
    # wind_direction_reprojected = direction_modes_sfc.rio.reproject_match(
    #     blurred_bp, resampling=Resampling.nearest
    # ).rename({'y': 'latitude', 'x': 'longitude'})
    # wind_direction_reprojected = assign_crs(wind_direction_reprojected, crs='EPSG:4326')
    # import xarray.testing as xrt
    # xrt.assert_equal(wind_direction_reprojected, wind_direction_reprojected_xr)

    # Adding int coercion because rasterio outputs ints, while scipy/xarray outputs floats.
    wind_direction_reprojected = direction_modes_sfc.interp_like(
        blurred_bp, method='nearest'
    ).astype(int)

    wind_informed_bp = create_composite_bp_map(blurred_bp, wind_direction_reprojected).drop_vars(
        'direction'
    )
    # Fix tiny FP misalignment in .sel of lat/lon between two datasets.
    wind_informed_bp_float_corrected = wind_informed_bp.assign_coords(
        latitude=rps_30_subset.latitude, longitude=rps_30_subset.longitude
    )
    return wind_informed_bp_float_corrected


def calculate_wind_adjusted_risk(region_id: str) -> xr.Dataset:
    from odc.geo.xr import assign_crs

    from ocr import catalog
    from ocr.chunking_config import ChunkingConfig
    from ocr.utils import lon_to_180

    console.log(f'Calculating wind risk for region: {region_id}')

    config = ChunkingConfig()

    # Open input dataset: USFS 30m community risk, USFS 30m interpolated 2011 climate runs and 1/4 degree? ERA5 Wind.
    climate_run_2011 = catalog.get_dataset('2011-climate-run-30m-4326').to_xarray()[['BP']]
    climate_run_2047 = catalog.get_dataset('2047-climate-run-30m-4326').to_xarray()[['BP']]

    rps_30 = catalog.get_dataset('USFS-wildfire-risk-communities-4326').to_xarray()[
        ['BP', 'CRPS', 'RPS']
    ]
    important_days = catalog.get_dataset('era5-fire-weather-days').to_xarray()[['sfcWindfromdir']]
    # TODO: Input datasets should already be pre-processed, so this transform should be done upstream.
    important_days = lon_to_180(important_days)

    y_slice, x_slice = config.region_id_to_latlon_slices(region_id=region_id)
    rps_30_subset = rps_30.sel(latitude=y_slice, longitude=x_slice)
    climate_run_2011_subset = climate_run_2011.sel(latitude=y_slice, longitude=x_slice).chunk(
        {'latitude': 6000, 'longitude': 4500}
    )
    climate_run_2047_subset = climate_run_2047.sel(latitude=y_slice, longitude=x_slice).chunk(
        {'latitude': 6000, 'longitude': 4500}
    )

    # Since important_days / wind is a lower resolution (.25 degrees?), we add in spatial buffer to match the resolution.
    wind_res = 0.25
    buffer = wind_res * 2  # add in a 2x buffer of the resolution
    buffered_y_slice = slice(y_slice.start + buffer, y_slice.stop - buffer, y_slice.step)
    buffered_x_slice = slice(x_slice.start - buffer, x_slice.stop + buffer, x_slice.step)

    wind_directions = important_days.sel(latitude=buffered_y_slice, longitude=buffered_x_slice)
    direction_indices = classify_wind_directions(wind_directions).chunk(dict(time=-1))
    direction_modes = apply_mode_calc(direction_indices).compute()

    direction_modes_sfc = assign_crs(direction_modes['sfcWindfromdir'], crs='EPSG:4326')

    wind_informed_bp_float_corrected_2011 = classify_wind(
        climate_run_subset=climate_run_2011_subset,
        direction_modes_sfc=direction_modes_sfc,
        rps_30_subset=rps_30_subset,
    )
    wind_informed_bp_float_corrected_2047 = classify_wind(
        climate_run_subset=climate_run_2047_subset,
        direction_modes_sfc=direction_modes_sfc,
        rps_30_subset=rps_30_subset,
    )

    # Add in non-wind-adjusted 2011 and 2047 BP*CRPS score for QA comparison

    climate_run_2011_subset_float_corrected = climate_run_2011_subset.assign_coords(
        latitude=wind_informed_bp_float_corrected_2011.latitude,
        longitude=wind_informed_bp_float_corrected_2011.longitude,
    )
    climate_run_2047_subset_float_corrected = climate_run_2047_subset.assign_coords(
        latitude=wind_informed_bp_float_corrected_2047.latitude,
        longitude=wind_informed_bp_float_corrected_2047.longitude,
    )

    risk_4326_combined = (
        climate_run_2011_subset_float_corrected['BP'] * rps_30_subset['CRPS']
    ).to_dataset(name='risk_2011')

    risk_4326_combined['risk_2047'] = (
        climate_run_2047_subset_float_corrected['BP'] * rps_30_subset['CRPS']
    )

    # Adjust USFS 30m CRPS with wind informed burn probability
    risk_4326_combined['wind_risk_2011'] = (
        wind_informed_bp_float_corrected_2011 * rps_30_subset['CRPS']
    )
    risk_4326_combined['wind_risk_2047'] = (
        wind_informed_bp_float_corrected_2047 * rps_30_subset['CRPS']
    )

    return risk_4326_combined.drop_vars(['spatial_ref'])
