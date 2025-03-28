import typing

import cv2 as cv
import numpy as np
import xarray as xr
from scipy.ndimage import rotate


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

    # TODO: @orianac, do we need to re-normalize the weights here to ensure sum equals 1.0?
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
    # TODO: add the non-directional spread
    weights_dict = generate_wind_directional_kernels(
        kernel_size=kernel_size, circle_diameter=circle_diameter
    )
    # do the spreading in each of the 8 directions with the correct weights
    # TODO: @orianac, do we want to support dask arrays here?
    # initialize dataset with the original burn probability
    spread_results = xr.Dataset(data_vars = {var_name: (da.dims, da.values) for var_name in weights_dict.keys()}, coords=da.coords)
    for direction, weights in weights_dict.items():  
        # spread_results[direction] = xr.zeros_like(da)
        for i in np.arange(
            iterations
        ):  # TODO, @orianac, is there a reason we are iterating over (iterations) without using the index. It appears that the output is the same regardless of the number of iterations.
            spread_results[direction] = (da.dims, cv.filter2D(spread_results[direction].values, -1, weights))
    return spread_results
