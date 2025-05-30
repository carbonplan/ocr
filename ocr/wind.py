import typing

import numpy as np
import xarray as xr


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


def create_finescale_wind_direction(bp: xr.Dataset, wind_direction: xr.Dataset) -> xr.Dataset:
    from rasterio.warp import Resampling

    wind_direction = wind_direction.rio.write_crs('EPSG:4326')
    bp = bp.rio.write_crs('EPSG:5070')
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
