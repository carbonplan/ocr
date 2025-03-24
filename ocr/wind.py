import typing

import numpy as np


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
