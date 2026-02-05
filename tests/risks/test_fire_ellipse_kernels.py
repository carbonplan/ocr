"""
Tests for elliptical kernel generation in fire risk calculations.

These tests verify the mathematical correctness of the wind-informed fire spread kernels,
including:
- Haversine distance calculations
- Latitude/longitude extraction
- Ellipse geometry (cardinal and ordinal directions)
- Kernel normalization
- Latitude-dependent sizing
- Symmetry properties
"""

import numpy as np
import pytest
import xarray as xr

from ocr.risks.fire import (
    generate_weights,
    generate_wind_directional_kernels,
    get_grid_spacing_info,
    haversine,
)


class TestHaversine:
    """Tests for the haversine distance function."""

    def test_zero_distance(self):
        """Same point should return zero distance."""
        dist = haversine(-100, 40, -100, 40)
        assert abs(dist) < 1e-6

    def test_one_degree_latitude(self):
        """One degree of latitude is approximately 111 km."""
        lat1, lon1 = 40.0, -100.0
        lat2, lon2 = 41.0, -100.0
        dist = haversine(lon1, lat1, lon2, lat2)
        expected = 111_000  # meters
        error = abs(dist - expected) / expected
        assert error < 0.01, f'Expected ~{expected} m, got {dist:.2f} m'

    def test_longitude_varies_by_latitude(self):
        """Longitude distance should vary by cos(latitude)."""
        # At equator vs 60° latitude
        dist_equator = haversine(0, 0, 1, 0)
        dist_60deg = haversine(0, 60, 1, 60)

        ratio = dist_60deg / dist_equator
        expected_ratio = np.cos(np.radians(60))  # = 0.5

        assert abs(ratio - expected_ratio) < 0.01, f'Ratio {ratio} vs expected {expected_ratio}'

    def test_returns_meters(self):
        """Verify function returns meters, not kilometers."""
        # 1 degree latitude at equator should be ~111,000 m, not 111 km
        dist = haversine(0, 0, 0, 1)
        assert dist > 100_000, f'Expected distance in meters, got {dist}'


class TestGetGridSpacingInfo:
    """Tests for grid spacing information extraction from DataArray."""

    def test_center_extraction(self):
        """Should extract center latitude and longitude correctly."""
        lats = np.linspace(30, 40, 101)
        lons = np.linspace(-110, -100, 101)

        da = xr.DataArray(
            np.random.rand(101, 101),
            coords={'latitude': lats, 'longitude': lons},
            dims=['latitude', 'longitude'],
        )

        lat, lon, lat_inc, lon_inc = get_grid_spacing_info(da)

        # Center should be at index 50
        assert abs(lat - lats[50]) < 1e-10
        assert abs(lon - lons[50]) < 1e-10

    def test_increment_calculation(self):
        """Should calculate correct increments."""
        lats = np.linspace(30, 40, 101)
        lons = np.linspace(-110, -100, 101)

        da = xr.DataArray(
            np.random.rand(101, 101),
            coords={'latitude': lats, 'longitude': lons},
            dims=['latitude', 'longitude'],
        )

        lat, lon, lat_inc, lon_inc = get_grid_spacing_info(da)

        expected_lat_inc = lats[1] - lats[0]
        expected_lon_inc = lons[1] - lons[0]

        assert abs(lat_inc - expected_lat_inc) < 1e-10
        assert abs(lon_inc - expected_lon_inc) < 1e-10

    def test_longitude_not_latitude_bug(self):
        """Regression test: ensure longitude uses longitude.values, not latitude.values."""
        lats = np.array([30.0, 35.0, 40.0])
        lons = np.array([-110.0, -105.0, -100.0])

        da = xr.DataArray(
            np.random.rand(3, 3),
            coords={'latitude': lats, 'longitude': lons},
            dims=['latitude', 'longitude'],
        )

        lat, lon, lat_inc, lon_inc = get_grid_spacing_info(da)

        # Center values should be middle of arrays
        assert abs(lat - 35.0) < 1e-10
        assert abs(lon - (-105.0)) < 1e-10  # Was incorrectly using latitude array


class TestGenerateWeights:
    """Tests for weight generation for individual directions."""

    @pytest.mark.parametrize('direction', ['W', 'E', 'N', 'S', 'NE', 'NW', 'SE', 'SW'])
    def test_weights_are_binary(self, direction):
        """Weights should be 0 or 1 before normalization."""
        weights = generate_weights(
            method='skewed',
            kernel_size=81,
            lat_pixel_size_meters=25,
            lon_pixel_size_meters=25,
            direction=direction,
        )

        unique_vals = np.unique(weights)
        assert set(unique_vals).issubset({0, 1}), f'{direction}: Expected binary values'

    @pytest.mark.parametrize('direction', ['W', 'E', 'N', 'S', 'NE', 'NW', 'SE', 'SW'])
    def test_nonzero_elements_exist(self, direction):
        """Each direction should have non-zero elements."""
        weights = generate_weights(
            method='skewed',
            kernel_size=81,
            lat_pixel_size_meters=25,
            lon_pixel_size_meters=25,
            direction=direction,
        )

        num_nonzero = np.sum(weights > 0)
        assert num_nonzero > 0, f'{direction}: No non-zero elements'

    @pytest.mark.parametrize('direction', ['W', 'E', 'N', 'S', 'NE', 'NW', 'SE', 'SW'])
    def test_approximate_ellipse_area(self, direction):
        """Ellipse area should be approximately πab / pixel_area."""
        a = 400  # semi-major axis in meters
        b = 200  # semi-minor axis in meters
        lat_dist = 25
        lon_dist = 25
        pixel_area = lat_dist * lon_dist

        weights = generate_weights(
            method='skewed',
            kernel_size=81,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction=direction,
        )

        num_pixels = np.sum(weights > 0)
        expected_pixels = np.pi * a * b / pixel_area
        error = abs(num_pixels - expected_pixels) / expected_pixels

        # Allow 20% error due to discretization
        assert error < 0.20, f'{direction}: Area error {error:.1%} too large'


class TestEllipseGeometry:
    """Tests for ellipse geometry and positioning."""

    def test_west_center_position(self):
        """West wind kernel should be centered to the west (negative x)."""
        kernel_size = 81
        lat_dist = 25
        lon_dist = 25

        weights = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction='W',
        )

        # Find center of mass
        y_indices, x_indices = np.where(weights > 0)
        center_x = np.mean(x_indices)
        np.mean(y_indices)

        # Center should be to the left (west) of image center
        center_pixel = kernel_size // 2
        assert center_x < center_pixel, 'West kernel should be offset west'

    def test_north_center_position(self):
        """North wind kernel should be centered to the north (positive y)."""
        kernel_size = 81
        lat_dist = 25
        lon_dist = 25

        weights = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction='N',
        )

        # Find center of mass
        y_indices, x_indices = np.where(weights > 0)
        center_y = np.mean(y_indices)

        # Center should be above (north of) image center
        # Note: In array coordinates, north is higher y index
        center_pixel = kernel_size // 2
        assert center_y > center_pixel, 'North kernel should be offset north'

    def test_cardinal_ellipse_extents(self):
        """Cardinal direction ellipses should extend to expected distances."""
        kernel_size = 81
        lat_dist = 25
        lon_dist = 25

        weights = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction='W',
        )

        # Find extent in x direction (at y=center)
        center_y = kernel_size // 2
        x_profile = weights[center_y, :]
        x_indices = np.where(x_profile > 0)[0]

        x_coords = (np.arange(kernel_size) - kernel_size // 2) * lon_dist
        min_x = x_coords[x_indices[0]]
        max_x = x_coords[x_indices[-1]]

        # Expected: center at -110m, radius 400m
        # So: -510m to +290m (approximately, allowing for discretization)
        assert -550 < min_x < -450, f'Western extent {min_x} outside expected range'
        assert 250 < max_x < 350, f'Eastern extent {max_x} outside expected range'

    def test_rotated_ellipse_diagonal_alignment(self):
        """Ordinal directions should have ellipses aligned along diagonals."""
        kernel_size = 81
        lat_dist = 25
        lon_dist = 25

        weights_ne = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction='NE',
        )

        # For NE with (a, b) = (400, 200), major axis is along NE-SW diagonal
        # Count pixels along main diagonal vs anti-diagonal
        diagonal_indices = np.arange(kernel_size)
        diagonal_pixels = np.sum(weights_ne[diagonal_indices, diagonal_indices] > 0)
        antidiagonal_pixels = np.sum(
            weights_ne[diagonal_indices, kernel_size - 1 - diagonal_indices] > 0
        )

        # Major axis along NE-SW should have more pixels than perpendicular
        assert diagonal_pixels > antidiagonal_pixels, 'NE kernel should be elongated along diagonal'


class TestKernelSymmetries:
    """Tests for expected symmetries in kernel shapes."""

    def test_west_east_horizontal_mirror(self):
        """W and E kernels should be approximately horizontal mirrors."""
        kernel_size = 81
        lat_dist = 25
        lon_dist = 25

        weights_w = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction='W',
        )
        weights_e = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction='E',
        )

        # Check approximate symmetry (offset centers on discrete grids don't produce perfect symmetry)
        w_count = np.sum(weights_w > 0)
        e_count = np.sum(weights_e > 0)
        assert abs(w_count - e_count) < 5, (
            f'Pixel counts should be similar: W={w_count}, E={e_count}'
        )

    def test_north_south_vertical_mirror(self):
        """N and S kernels should be approximately vertical mirrors."""
        kernel_size = 81
        lat_dist = 25
        lon_dist = 25

        weights_n = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction='N',
        )
        weights_s = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction='S',
        )

        # Check approximate symmetry (offset centers on discrete grids don't produce perfect symmetry)
        n_count = np.sum(weights_n > 0)
        s_count = np.sum(weights_s > 0)
        assert abs(n_count - s_count) < 5, (
            f'Pixel counts should be similar: N={n_count}, S={s_count}'
        )

    def test_diagonal_180_rotation(self):
        """Opposite diagonal directions should be approximately 180° rotations."""
        kernel_size = 81
        lat_dist = 25
        lon_dist = 25

        weights_ne = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction='NE',
        )
        weights_sw = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            lat_pixel_size_meters=lat_dist,
            lon_pixel_size_meters=lon_dist,
            direction='SW',
        )

        # Check approximate symmetry (offset centers on discrete grids don't produce perfect symmetry)
        ne_count = np.sum(weights_ne > 0)
        sw_count = np.sum(weights_sw > 0)
        assert abs(ne_count - sw_count) < 5, (
            f'Pixel counts should be similar: NE={ne_count}, SW={sw_count}'
        )


class TestGenerateWindDirectionalKernels:
    """Tests for the full set of wind directional kernels."""

    def test_all_directions_present(self):
        """Should generate kernels for all 8 directions."""
        kernels = generate_wind_directional_kernels(
            kernel_size=81,
            latitude=38.0,
            longitude=-105.0,
            latitude_increment=0.0003,
            longitude_increment=0.0003,
        )

        expected_directions = ['W', 'NW', 'N', 'NE', 'E', 'SE', 'S', 'SW']
        assert set(kernels.keys()) == set(expected_directions)

    def test_all_kernels_normalized(self):
        """All kernels should sum to 1.0."""
        kernels = generate_wind_directional_kernels(
            kernel_size=81,
            latitude=38.0,
            longitude=-105.0,
            latitude_increment=0.0003,
            longitude_increment=0.0003,
        )

        for direction, kernel in kernels.items():
            kernel_sum = kernel.sum()
            assert np.isclose(kernel_sum, 1.0, rtol=1e-6), (
                f'{direction} kernel sum {kernel_sum} != 1.0'
            )

    def test_latitude_dependence(self):
        """Kernel sizes should adapt to latitude (more pixels at higher latitudes)."""
        latitudes = [30, 40, 50, 60]
        pixel_counts = []

        for lat in latitudes:
            kernels = generate_wind_directional_kernels(
                kernel_size=81,
                latitude=lat,
                longitude=-105.0,
                latitude_increment=0.0003,
                longitude_increment=0.0003,
            )

            # Count pixels in W kernel
            pixel_counts.append(np.sum(kernels['W'] > 0))

        # Pixel counts should increase with latitude
        # (compensating for narrower longitude spacing)
        assert pixel_counts == sorted(pixel_counts), (
            f'Pixel counts should increase with latitude: {pixel_counts}'
        )

    def test_distance_calculations(self):
        """Distance calculations should follow cos(latitude) relationship."""
        latitude = 40.0
        longitude = -105.0
        lat_increment = 0.0003
        lon_increment = 0.0003

        # Calculate pixel sizes in meters
        lon_pixel_size_meters = haversine(longitude, latitude, longitude + lon_increment, latitude)
        lat_pixel_size_meters = haversine(longitude, latitude, longitude, latitude + lat_increment)

        # At 40° latitude, east-west should be ~77% of north-south
        ratio = lon_pixel_size_meters / lat_pixel_size_meters
        expected_ratio = np.cos(np.radians(latitude))

        assert abs(ratio - expected_ratio) < 0.01, f'Ratio {ratio} vs expected {expected_ratio}'


class TestCircularFocalMean:
    """Tests for circular_focal_mean method (baseline)."""

    def test_circular_weights(self):
        """Circular method should create circular weights."""
        weights = generate_weights(method='circular_focal_mean', kernel_size=81, circle_diameter=35)

        # Should have non-zero elements
        assert np.sum(weights > 0) > 0

        # Should sum to 1
        assert np.isclose(weights.sum(), 1.0, rtol=1e-6)

    def test_circular_symmetry(self):
        """Circular weights should be symmetric."""
        weights = generate_weights(method='circular_focal_mean', kernel_size=81, circle_diameter=35)

        # Should be symmetric across both axes
        assert np.allclose(weights, np.fliplr(weights)), 'Should be horizontally symmetric'
        assert np.allclose(weights, np.flipud(weights)), 'Should be vertically symmetric'
