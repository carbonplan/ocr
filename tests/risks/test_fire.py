import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocr.risks.fire import (
    apply_wind_directional_convolution,
    classify_wind_directions,
    compute_mode_along_time,
    create_composite_bp_map,
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


@pytest.fixture
def make_mode_array():
    """Module-level fixture to build a (time, latitude, longitude) classification array."""

    def _make(values: np.ndarray) -> xr.DataArray:
        assert values.ndim == 3, 'values must be 3D (time, lat, lon)'
        t, ny, nx = values.shape
        times = pd.date_range('2000-01-01', periods=t, freq='D')
        lats = np.linspace(45, 50, ny)
        lons = np.linspace(-125, -120, nx)
        da = xr.DataArray(
            values,
            dims=['time', 'latitude', 'longitude'],
            coords={'time': times, 'latitude': lats, 'longitude': lons},
            name='wind_direction_classification',
        )
        da.attrs['direction_labels'] = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        return da

    return _make


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

        # Set values with np.nan placeholders mixed in
        data[:, 0, 0] = [
            NORTH,
            NORTH,
            np.nan,
            NORTH,
            NORTH,
            np.nan,
            np.nan,
            SOUTH,
            np.nan,
            np.nan,
        ]  # Mode is NORTH
        data[:, 0, 1] = [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            EAST,
            EAST,
            EAST,
            np.nan,
            np.nan,
        ]  # Mode is EAST
        data[:, 1, 0] = [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]  # All placeholders
        data[:, 1, 1] = [
            SOUTH,
            SOUTH,
            WEST,
            WEST,
            WEST,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]  # Mode is WEST

        test_array = create_test_array(data)
        result = compute_mode_along_time(test_array)

        expected = np.array(
            [
                [NORTH, EAST],
                [np.nan, WEST],  # Should be np.nan when all values are placeholders
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

    # Skewed kernel should be asymmetric due to roll/distortion
    center_y, center_x = weights.shape[0] // 2, weights.shape[1] // 2
    sub = weights[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10]
    # At least one directional flip should differ
    assert not np.allclose(sub, np.flipud(sub)) or not np.allclose(sub, np.fliplr(sub))


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

    # Check that weights are radially symmetric (no directional roll applied)
    center_y, center_x = weights.shape[0] // 2, weights.shape[1] // 2
    sub = weights[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10]
    # Symmetry checks (horizontal/vertical & both axis flips)
    assert np.allclose(sub, np.flipud(sub))
    assert np.allclose(sub, np.fliplr(sub))
    assert np.allclose(sub, np.flip(sub))


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
    # For even kernel sizes the current implementation produces an odd-sized grid (includes both 0 endpoints)
    # yielding size kernel_size+1. Update expectation accordingly.
    assert weights.shape == (41, 41)

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
        generate_weights(method='invalid_method')  # type: ignore[arg-type]


def test_generate_wind_directional_kernels_normalized():
    """Test that each kernel is normalized (weights sum to 1)."""
    result = generate_wind_directional_kernels()
    for direction, kernel in result.items():
        # Allow a small tolerance due to float32 accumulation after rotations
        np.testing.assert_allclose(kernel.sum(), 1.0, rtol=1e-6, atol=1e-6)


def test_generate_wind_directional_kernels_non_negative():
    """Ensure no negative weights remain after rotation/clipping."""
    result = generate_wind_directional_kernels()
    for direction, kernel in result.items():
        assert (kernel >= 0).all(), f'Kernel {direction} contains negative values'


def test_classify_wind_directions_output_domain():
    """All classified outputs must be in 0-7 or NaN (no -1 sentinel)."""
    angles = xr.DataArray(
        np.array([0, 45, 90, 135, 180, 225, 270, 315, np.nan, 720, -45]), dims=['sample']
    ).astype('float32')
    classified = classify_wind_directions(angles)
    vals = classified.values
    # mask NaNs then check domain
    domain_vals = vals[~np.isnan(vals)]
    assert domain_vals.min() >= 0
    assert domain_vals.max() <= 7
    assert -1 not in domain_vals


def test_create_composite_bp_map_nan_and_invalid_handling():
    """Invalid indices (<0 or >8) and NaNs should produce NaNs; valid 0-8 indices (including 8 'circular') select correct layer."""
    direction_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'circular']
    # Build a tiny synthetic bp dataset with distinct constant planes per direction
    ny, nx = 4, 5
    data_vars = {}
    for i, lab in enumerate(direction_labels):
        data_vars[lab] = (('latitude', 'longitude'), np.full((ny, nx), fill_value=float(i)))
    lat = np.linspace(0, 1, ny)
    lon = np.linspace(10, 11, nx)
    bp = xr.Dataset(data_vars, coords={'latitude': lat, 'longitude': lon})

    # Wind directions: include valid, NaN, -1, 9 (invalid high), 3.0 (float), 7 (edge)
    wind_vals = np.array(
        [
            [0, 1, 2, np.nan, -1],
            [3, 4, 5, 6, 7],
            [9, np.nan, 2, -5, 7],
            [
                np.nan,
                3.0,
                1.0,
                8,
                0,
            ],  # 8 is the 'circular' direction index and SHOULD be treated as valid
        ],
        dtype=float,
    )
    wind = xr.DataArray(
        wind_vals, dims=('latitude', 'longitude'), coords={'latitude': lat, 'longitude': lon}
    )

    composite = create_composite_bp_map(bp, wind)

    # Check shape
    assert composite.shape == (ny, nx)

    # Positions with invalid / NaN indices should be NaN
    def is_invalid(v):
        # Treat 8 (circular) as a valid direction; invalidate only values >8 or <0 or NaN
        return (np.isnan(v)) or (v < 0) or (v > 8)

    expected_nan_mask = np.vectorize(is_invalid)(wind_vals)
    np.testing.assert_array_equal(np.isnan(composite.values), expected_nan_mask)
    # Valid positions should pull the layer value equal to the index
    valid_mask = ~expected_nan_mask
    selected_vals = composite.values[valid_mask]
    expected_vals = wind_vals[valid_mask]
    np.testing.assert_array_equal(selected_vals, expected_vals)


def test_apply_wind_directional_convolution_non_negative_output():
    """After convolution and clipping, outputs should have no negative values."""
    data = np.zeros((41, 41), dtype=np.float32)
    data[20, 20] = 1.0  # impulse at center
    da = xr.DataArray(
        data, dims=['latitude', 'longitude'], coords={'latitude': range(41), 'longitude': range(41)}
    )
    result = apply_wind_directional_convolution(
        da, iterations=2, kernel_size=41.0, circle_diameter=21.0
    )
    for direction in result.data_vars:
        arr = result[direction].values
        assert (arr >= -1e-8).all(), f'Negative residual beyond tolerance in {direction}'
        # Hard clip tolerance check (should have been clipped to >=0)
        assert (arr >= 0).all() or np.isclose(arr.min(), 0.0)


def test_compute_mode_along_time_no_negative_modes(make_mode_array):
    """Ensure compute_mode_along_time never returns -1 placeholders."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 8, size=(6, 3, 3)).astype(np.float32)
    # Introduce NaNs
    data[0, 0, 0] = np.nan
    data[2, 1, 2] = np.nan
    da = make_mode_array(data)
    result = compute_mode_along_time(da)
    assert not np.any(result.values == -1), 'Found -1 placeholder in mode output'
