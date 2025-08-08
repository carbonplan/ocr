import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocr.risks.fire import (
    apply_wind_directional_convolution,
    classify_wind_directions,
    compute_mode_along_time,
    generate_angles,
    generate_weights,
    generate_wind_directional_kernels,
)

############################################
# Tests for wind direction classification ##
############################################

NORTH = 0
NORTHEAST = 1
EAST = 2
SOUTHEAST = 3
SOUTH = 4
SOUTHWEST = 5
WEST = 6
NORTHWEST = 7


class TestWindDirectionClassification:
    @pytest.fixture
    def create_test_data(self):
        """Create test wind direction data as xarray DataArray"""

        def _create(values):
            if isinstance(values, int | float):
                values = np.array([values])
            return xr.DataArray(values).astype('float32')

        return _create

    @pytest.mark.parametrize(
        'angle,expected_class',
        [
            # Test center of each bin
            (0, NORTH),  # North center
            (45, NORTHEAST),  # Northeast center
            (90, EAST),  # East center
            (135, SOUTHEAST),  # Southeast center
            (180, SOUTH),  # South center
            (225, SOUTHWEST),  # Southwest center
            (270, WEST),  # West center
            (315, NORTHWEST),  # Northwest center
            # Test exact boundary cases
            (22.5, NORTHEAST),  # N-NE boundary
            (67.5, EAST),  # NE-E boundary
            (112.5, SOUTHEAST),  # E-SE boundary
            (157.5, SOUTH),  # SE-S boundary
            (202.5, SOUTHWEST),  # S-SW boundary
            (247.5, WEST),  # SW-W boundary
            (292.5, NORTHWEST),  # W-NW boundary
            (337.5, NORTH),  # NW-N boundary
            # Test near-boundary cases
            (22.4, NORTH),
            (22.6, NORTHEAST),
            (337.4, NORTHWEST),
            (337.6, NORTH),
            # Test critical North edge case (near 360)
            (359.9, NORTH),
            (0.1, NORTH),
            (360, NORTH),
        ],
    )
    def test_direction_classification(self, angle, expected_class, create_test_data):
        """Test classification for various wind directions."""
        wind_dir = create_test_data(angle)
        result = classify_wind_directions(wind_dir).values[0]
        assert result == expected_class, f'Expected {expected_class} for {angle}°, got {result}'

    def test_north_edge_case(self, create_test_data):
        """Specifically test the North direction edge case."""
        angles = [0, 10, 20, 340, 345, 350, 355, 359, 360]
        wind_dir = create_test_data(angles)
        result = classify_wind_directions(wind_dir).values
        expected = np.array([NORTH] * len(angles))
        np.testing.assert_array_equal(result, expected)

    def test_values_outside_range(self, create_test_data):
        """Test handling of values outside the 0-360 range."""
        # Values outside standard range should wrap around properly
        angles = [-45, 405, 720, -315]  # Equivalent to 315, 45, 0, 45
        expected = [NORTHWEST, NORTHEAST, NORTH, NORTHEAST]
        wind_dir = create_test_data(angles)
        result = classify_wind_directions(wind_dir).values
        np.testing.assert_array_equal(result, expected)

    def test_nan_handling(self, create_test_data):
        """Test handling of NaN values."""
        angles = [0, np.nan, 90, np.nan, 180]
        wind_dir = create_test_data(angles)
        result = classify_wind_directions(wind_dir).values
        expected = np.array([NORTH, np.nan, EAST, np.nan, SOUTH])
        np.testing.assert_array_equal(result, expected)


class TestComputeModeAlongTime:
    @pytest.fixture
    def create_test_array(self):
        """Create a test DataArray with time, latitude, and longitude dimensions"""

        def _create(values, chunks=None):
            # Default shape if values is not already an array
            if not isinstance(values, np.ndarray):
                values = np.array([values])

            # Create the data array
            if chunks:
                data = da.from_array(values, chunks=chunks)
            else:
                data = values

            # If it's a 3D array, assume it has time, lat, lon dimensions
            if values.ndim == 3:
                times = pd.date_range('2000-01-01', periods=values.shape[0], freq='D')
                lats = np.linspace(45, 50, values.shape[1])
                lons = np.linspace(-125, -120, values.shape[2])

                result = xr.DataArray(
                    data,
                    dims=['time', 'latitude', 'longitude'],
                    coords={'time': times, 'latitude': lats, 'longitude': lons},
                    name='wind_direction_classification',
                )
            else:
                # For 1D arrays, just use time dimension
                times = pd.date_range('2000-01-01', periods=len(values), freq='D')
                result = xr.DataArray(
                    data,
                    dims=['time'],
                    coords={'time': times},
                    name='wind_direction_classification',
                )

            # Add attributes
            result.attrs['direction_labels'] = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

            return result

        return _create

    def test_basic_mode_calculation(self, create_test_array):
        """Test that the function correctly calculates the mode along the time dimension."""
        # Create data with a clear mode at each point
        data = np.zeros((10, 2, 2), dtype=np.float32)

        # Set different modes for each lat/lon point
        data[:, 0, 0] = [
            NORTH,
            NORTH,
            NORTH,
            EAST,
            EAST,
            EAST,
            EAST,
            SOUTH,
            SOUTH,
            SOUTH,
        ]  # Mode is EAST (2)
        data[:, 0, 1] = [
            NORTH,
            NORTH,
            NORTH,
            NORTH,
            NORTH,
            SOUTH,
            SOUTH,
            SOUTH,
            SOUTH,
            WEST,
        ]  # Mode is NORTH (0)
        data[:, 1, 0] = [
            SOUTH,
            SOUTH,
            SOUTH,
            SOUTH,
            SOUTH,
            SOUTH,
            WEST,
            WEST,
            WEST,
            WEST,
        ]  # Mode is SOUTH (4)
        data[:, 1, 1] = [
            WEST,
            WEST,
            WEST,
            WEST,
            WEST,
            WEST,
            WEST,
            WEST,
            WEST,
            WEST,
        ]  # Mode is WEST (6)

        test_array = create_test_array(data)
        result = compute_mode_along_time(test_array)

        expected = np.array([[EAST, NORTH], [SOUTH, WEST]], dtype=np.float32)

        np.testing.assert_array_equal(result.values, expected)

    def test_nan_handling(self, create_test_array):
        """Test handling of NaN values."""
        # Create data with NaN values
        data = np.full((10, 2, 2), np.nan, dtype=np.float32)

        # Set values with NaNs mixed in
        data[0:5, 0, 0] = [NORTH, NORTH, np.nan, NORTH, NORTH]  # Mode is NORTH
        data[0:5, 0, 1] = [EAST, np.nan, np.nan, EAST, EAST]  # Mode is EAST
        data[0:5, 1, 0] = [np.nan, np.nan, np.nan, np.nan, np.nan]  # All NaNs
        data[0:5, 1, 1] = [SOUTH, SOUTH, WEST, WEST, WEST]  # Mode is WEST

        test_array = create_test_array(data)
        result = compute_mode_along_time(test_array)

        expected = np.array([[NORTH, EAST], [np.nan, WEST]], dtype=np.float32)

        np.testing.assert_array_equal(result.values, expected)

    def test_placeholder_handling(self, create_test_array):
        """Test handling of placeholder values (-1)."""
        # Create data with placeholder values
        data = np.zeros((10, 2, 2), dtype=np.float32)

        # Set values with -1 placeholders mixed in
        data[:, 0, 0] = [NORTH, NORTH, -1, NORTH, NORTH, -1, -1, SOUTH, -1, -1]  # Mode is NORTH
        data[:, 0, 1] = [-1, -1, -1, -1, -1, EAST, EAST, EAST, -1, -1]  # Mode is EAST
        data[:, 1, 0] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  # All placeholders
        data[:, 1, 1] = [SOUTH, SOUTH, WEST, WEST, WEST, -1, -1, -1, -1, -1]  # Mode is WEST

        test_array = create_test_array(data)
        result = compute_mode_along_time(test_array)

        expected = np.array(
            [
                [NORTH, EAST],
                [np.nan, WEST],  # Should be NaN when all values are placeholders
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(result.values, expected)

    @pytest.mark.xfail(reason='Ties are not yet handled')
    def test_tie_handling(self, create_test_array):
        """Test the case where there's a tie for the mode."""

        data = np.zeros((10, 2, 1), dtype=np.float32)

        # Two modes with equal count
        data[:, 0, 0] = [NORTH, NORTH, NORTH, NORTH, NORTH, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH]
        # Three directions with equal count
        data[:, 1, 0] = [EAST, EAST, EAST, WEST, WEST, WEST, NORTH, NORTH, NORTH, SOUTH]

        test_array = create_test_array(data)
        result = compute_mode_along_time(test_array)

        # In case of tie, argmax returns the first occurrence
        # NORTH (0) should win in the first case because it appears first in the array
        # EAST (2) should win in the second case because it appears first
        expected = np.array([[NORTH], [EAST]], dtype=np.float32)

        np.testing.assert_array_equal(result.values, expected)

    def test_single_value(self, create_test_array):
        """Test with a single value in the time dimension."""
        data = np.array([[[NORTH, EAST], [SOUTH, WEST]]], dtype=np.float32)  # Just one time step

        test_array = create_test_array(data)
        result = compute_mode_along_time(test_array)

        expected = np.array([[NORTH, EAST], [SOUTH, WEST]], dtype=np.float32)

        np.testing.assert_array_equal(result.values, expected)

    def test_chunked_data(self, create_test_array):
        """Test with chunked dask arrays."""
        # Create data with a clear mode at each point
        data = np.zeros((20, 4, 4), dtype=np.float32)

        # Fill with various patterns
        for t in range(20):
            for y in range(4):
                for x in range(4):
                    # Create a pattern where the mode for each point is (y*4 + x) % 8
                    # This ensures different modes for different points
                    data[t, y, x] = (y * 4 + x + t) % 8

        # Override some points to create known modes
        data[:, 0, 0] = [NORTH] * 10 + [SOUTH] * 5 + [EAST] * 5  # Mode is NORTH
        data[:, 1, 1] = [WEST] * 15 + [EAST] * 5  # Mode is WEST

        # Create a chunked array
        chunks = (5, 2, 2)  # Time chunks of 5, space chunks of 2x2
        test_array = create_test_array(data, chunks=chunks)

        # Compute the mode
        result = compute_mode_along_time(test_array)

        # Verify the specific points we set
        assert result.values[0, 0] == NORTH
        assert result.values[1, 1] == WEST

        # Also check that the dimensions and coordinates are preserved
        assert result.dims == ('latitude', 'longitude')
        assert len(result.latitude) == 4
        assert len(result.longitude) == 4

    def test_metadata_preservation(self, create_test_array):
        """Test that metadata (coordinates, attributes) is preserved."""
        data = np.zeros((5, 3, 3), dtype=np.float32)
        # Fill with some values
        for t in range(5):
            data[t, :, :] = t % 8

        test_array = create_test_array(data)
        # Add some custom attributes
        test_array.attrs['custom_attr'] = 'test_value'

        result = compute_mode_along_time(test_array)

        # Check coordinates
        assert 'latitude' in result.coords
        assert 'longitude' in result.coords
        assert len(result.latitude) == 3
        assert len(result.longitude) == 3

        # Check attributes
        assert 'direction_labels' in result.attrs
        assert result.attrs['direction_labels'] == ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        assert 'custom_attr' in result.attrs
        assert result.attrs['custom_attr'] == 'test_value'
        assert 'long_name' in result.attrs  # New attribute added by the function

    def test_empty_array(self, create_test_array):
        """Test with an empty array (all NaNs)."""
        data = np.full((5, 2, 2), np.nan, dtype=np.float32)
        test_array = create_test_array(data)

        result = compute_mode_along_time(test_array)

        # All results should be NaN
        expected = np.full((2, 2), np.nan, dtype=np.float32)
        np.testing.assert_array_equal(
            result.values,
            expected,
        )


def test_generate_angles_returns_dict():
    """Test that generate_angles returns a dictionary."""
    result = generate_angles()
    assert isinstance(result, dict)


def test_generate_angles_has_correct_keys():
    """Test that generate_angles returns a dictionary with the expected direction keys."""
    result = generate_angles()
    expected_keys = {'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE', 'E'}
    assert set(result.keys()) == expected_keys


def test_generate_angles_has_correct_values():
    """Test that generate_angles returns the correct angle values."""
    result = generate_angles()
    # Check that all values are in the expected range
    for angle in result.values():
        assert 0 <= angle < 360
        assert isinstance(angle, np.float32)


def test_generate_angles_correct_mappings():
    """Test that directions map to the correct angles."""
    result = generate_angles()

    # Expected mappings based on the implementation
    expected_mappings = {
        'NE': 22.5,
        'N': 67.5,
        'NW': 112.5,
        'W': 157.5,
        'SW': 202.5,
        'S': 247.5,
        'SE': 292.5,
        'E': 337.5,
    }

    for direction, expected_angle in expected_mappings.items():
        assert np.isclose(result[direction], expected_angle)


def test_generate_angles_45_degree_intervals():
    """Test that angles are spaced at 45 degree intervals."""
    result = generate_angles()
    angles = sorted(result.values())

    # Check that differences between consecutive sorted angles are 45 degrees
    for i in range(len(angles) - 1):
        assert np.isclose(angles[i + 1] - angles[i], 45.0)


def test_generate_angles_float32_type():
    """Test that angles are of numpy.float32 type."""
    result = generate_angles()
    for angle in result.values():
        assert angle.dtype == np.float32


def test_generate_weights_defaults():
    """Test generate_weights with default parameters."""
    weights = generate_weights()

    # Check returned type
    assert isinstance(weights, np.ndarray)

    # Check shape with default kernel_size=81.0
    expected_shape = (81, 81)
    assert weights.shape == expected_shape

    # Check weights are normalized (sum to 1)
    assert np.isclose(weights.sum(), 1.0)


def test_generate_weights_skewed():
    """Test generate_weights with 'skewed' method."""
    weights = generate_weights(method='skewed', kernel_size=51.0, circle_diameter=25.0)

    # Check shape
    assert weights.shape == (51, 51)

    # Check normalization
    assert np.isclose(weights.sum(), 1.0)

    # Check that values are binary before normalization (0 outside circle, positive inside)
    unique_values = np.unique(weights * weights.sum())
    assert len(unique_values) == 2
    assert 0 in unique_values


def test_generate_weights_circular_focal_mean():
    """Test generate_weights with 'circular_focal_mean' method."""
    weights = generate_weights(method='circular_focal_mean', kernel_size=61.0, circle_diameter=29.0)

    # Check shape
    assert weights.shape == (61, 61)

    # Check normalization
    assert np.isclose(weights.sum(), 1.0)

    # Check that values are binary before normalization (0 outside circle, positive inside)
    unique_values = np.unique(weights * weights.sum())
    assert len(unique_values) == 2
    assert 0 in unique_values

    # Check that weights are rolled (should be asymmetric)
    center_y, center_x = weights.shape[0] // 2, weights.shape[1] // 2
    # Check if the roll was applied by verifying asymmetry
    assert not np.allclose(
        weights[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10],
        np.flip(weights[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10]),
    )


def test_generate_weights_odd_kernel_size():
    """Test with odd kernel sizes to check proper centering."""
    odd_kernel = 41
    weights = generate_weights(kernel_size=odd_kernel)

    # Check shape
    assert weights.shape == (41, 41)

    # Check center point is at the middle
    center_y, center_x = odd_kernel // 2, odd_kernel // 2

    # For 'skewed' method, the central region should have non-zero values
    assert weights[center_y, center_x] > 0


def test_generate_weights_even_kernel_size():
    """Test with even kernel sizes which might require special handling."""
    even_kernel = 40
    weights = generate_weights(kernel_size=even_kernel)

    # Check shape
    assert weights.shape == (40, 40)

    # The weight pattern should still be normalized
    assert np.isclose(weights.sum(), 1.0)


def test_generate_weights_small_circle():
    """Test with a small circle diameter."""
    weights = generate_weights(kernel_size=31.0, circle_diameter=5.0)

    # Count non-zero elements - should be small with a small circle
    non_zero = np.count_nonzero(weights)

    # The number of non-zero elements should be approximately π*(d/2)²
    # where d is the circle_diameter (plus possible edge effects)
    expected_count = np.pi * (5.0 / 2) ** 2
    # Allow some margin due to discretization
    assert non_zero < expected_count * 2

    # Check normalization
    assert np.isclose(weights.sum(), 1.0)


def test_generate_weights_large_circle():
    """Test with a large circle diameter relative to kernel size."""
    kernel_size = 41.0
    circle_diameter = 39.0  # Almost as big as the kernel

    weights = generate_weights(kernel_size=kernel_size, circle_diameter=circle_diameter)

    # Most of the kernel should be non-zero
    non_zero_fraction = np.count_nonzero(weights) / weights.size
    assert non_zero_fraction > 0.5

    # Check normalization
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_weights_positive():
    """Test that all weights are non-negative."""
    weights = generate_weights()
    assert np.all(weights >= 0)


def test_invalid_method():
    """Test that an invalid method raises an error."""
    with pytest.raises(ValueError):
        generate_weights(method='invalid_method')


def test_generate_wind_directional_kernels_normalized():
    """Test that each kernel is normalized (weights sum to 1)."""
    result = generate_wind_directional_kernels()
    for direction, kernel in result.items():
        np.testing.assert_allclose(kernel.sum(), 1.0)


@pytest.mark.xfail(
    reason='This test is currently failing due to a known issue with the kernel generation.'
)
def test_apply_wind_directional_convolution():
    data = np.zeros((21, 21))
    data[10, 10] = 1.0  # Single point in the center
    da = xr.DataArray(data, coords={'lat': range(21), 'lon': range(21)}, dims=['lat', 'lon'])
    result = apply_wind_directional_convolution(
        da, iterations=1, kernel_size=5.0, circle_diameter=3.0
    )

    # after convolution, the central point should be spread out. so we should have more than one non-zero value
    assert np.count_nonzero(result.data) > 1
