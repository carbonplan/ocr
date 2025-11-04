"""Snapshot tests for fire risk calculations.

These tests capture the full output of complex functions to detect
unintended changes in computation results across refactoring.
"""

import numpy as np
import pytest
import xarray as xr

from ocr.risks.fire import (
    apply_wind_directional_convolution,
    classify_wind_directions,
    compute_modal_wind_direction,
    compute_wind_direction_distribution,
    create_weighted_composite_bp_map,
    direction_histogram,
    fosberg_fire_weather_index,
    generate_wind_directional_kernels,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def sample_burn_probability():
    """Create a realistic burn probability array for testing."""
    ny, nx = 50, 50
    lat = np.linspace(35.0, 36.0, ny)
    lon = np.linspace(-120.0, -119.0, nx)

    # Create a gradient pattern with some features
    y_grid, x_grid = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    data = 0.001 + 0.01 * np.exp(-((y_grid - 25) ** 2 + (x_grid - 25) ** 2) / 100)

    return xr.DataArray(
        data.astype(np.float32),
        dims=['latitude', 'longitude'],
        coords={'latitude': lat, 'longitude': lon},
        name='BP',
    )


@pytest.fixture
def sample_wind_directions():
    """Create sample wind direction data."""
    ny, nx, nt = 10, 10, 100
    lat = np.linspace(35.0, 36.0, ny)
    lon = np.linspace(-120.0, -119.0, nx)
    time = np.arange(nt)

    # Create spatially varying wind directions with some patterns
    np.random.seed(42)
    base_direction = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315], size=(ny, nx))
    directions = np.zeros((nt, ny, nx))
    for t in range(nt):
        directions[t] = base_direction + np.random.normal(0, 10, size=(ny, nx))

    return xr.DataArray(
        directions.astype(np.float32) % 360,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': time, 'latitude': lat, 'longitude': lon},
        name='wind_direction',
    )


class TestWindDirectionalKernelsSnapshot:
    """Snapshot tests for wind directional kernels."""

    def test_generate_wind_directional_kernels_default(self, xarray_snapshot):
        """Snapshot test for default kernel generation."""
        kernels = generate_wind_directional_kernels()

        # Convert to Dataset for snapshot
        ds = xr.Dataset(
            {
                direction: xr.DataArray(
                    kernel,
                    dims=['y', 'x'],
                    coords={'y': range(kernel.shape[0]), 'x': range(kernel.shape[1])},
                )
                for direction, kernel in kernels.items()
            }
        )

        assert xarray_snapshot == ds

    def test_generate_wind_directional_kernels_custom(self, xarray_snapshot):
        """Snapshot test for custom kernel parameters."""
        kernels = generate_wind_directional_kernels(kernel_size=41.0, circle_diameter=21.0)

        ds = xr.Dataset(
            {
                direction: xr.DataArray(
                    kernel,
                    dims=['y', 'x'],
                    coords={'y': range(kernel.shape[0]), 'x': range(kernel.shape[1])},
                )
                for direction, kernel in kernels.items()
            }
        )

        assert xarray_snapshot == ds


class TestWindDirectionalConvolutionSnapshot:
    """Snapshot tests for wind directional convolution."""

    def test_apply_wind_directional_convolution_default(
        self, xarray_snapshot, sample_burn_probability
    ):
        """Snapshot test for directional convolution with default parameters."""
        result = apply_wind_directional_convolution(sample_burn_probability, iterations=3)
        assert xarray_snapshot == result

    def test_apply_wind_directional_convolution_single_iteration(
        self, xarray_snapshot, sample_burn_probability
    ):
        """Snapshot test for single iteration convolution."""
        result = apply_wind_directional_convolution(sample_burn_probability, iterations=1)
        assert xarray_snapshot == result


class TestWindClassificationSnapshot:
    """Snapshot tests for wind direction classification."""

    def test_classify_wind_directions_spatial(self, xarray_snapshot, sample_wind_directions):
        """Snapshot test for wind direction classification on spatial data."""
        # Take a single time slice
        wind_slice = sample_wind_directions.isel(time=0)
        result = classify_wind_directions(wind_slice).to_dataset()
        assert xarray_snapshot == result

    def test_classify_wind_directions_edge_cases(self, xarray_snapshot):
        """Snapshot test for edge cases in wind classification."""
        # Test values near boundaries and wrap-around
        angles = xr.DataArray(
            np.array(
                [
                    [0, 22.4, 22.6, 45, 67.5],
                    [90, 135, 180, 225, 270],
                    [315, 337.4, 337.6, 359.9, 360],
                    [np.nan, -45, 405, 720, -315],
                ],
                dtype=np.float32,
            ),
            dims=['y', 'x'],
            coords={'y': range(4), 'x': range(5)},
        )
        result = classify_wind_directions(angles).to_dataset()
        assert xarray_snapshot == result


class TestWindDistributionSnapshot:
    """Snapshot tests for wind direction distribution."""

    def test_direction_histogram(self, xarray_snapshot, sample_wind_directions):
        """Snapshot test for direction histogram computation."""
        classified = classify_wind_directions(sample_wind_directions)
        result = direction_histogram(classified).to_dataset()
        assert xarray_snapshot == result

    def test_compute_wind_direction_distribution(self, xarray_snapshot, sample_wind_directions):
        """Snapshot test for wind direction distribution with fire weather mask."""
        # Create a fire weather mask (True for high winds)
        fire_weather_mask = sample_wind_directions > 180

        result = compute_wind_direction_distribution(sample_wind_directions, fire_weather_mask)
        assert xarray_snapshot == result

    def test_compute_modal_wind_direction(self, xarray_snapshot):
        """Snapshot test for modal wind direction computation."""
        # Create a distribution with clear modes
        ny, nx = 5, 5
        distribution = xr.DataArray(
            np.array(
                [
                    [0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0],  # Mode at index 0 (N)
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Mode at index 4 (S)
                    [0.125] * 8,  # Uniform
                    [0.0] * 8,  # No fire weather
                    [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0],  # Tie - should pick first
                ]
                * nx,
                dtype=np.float32,
            )
            .reshape(ny, nx, 8)
            .transpose(2, 0, 1),
            dims=['wind_direction', 'latitude', 'longitude'],
            coords={
                'wind_direction': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                'latitude': range(ny),
                'longitude': range(nx),
            },
            name='wind_direction_distribution',
        )

        result = compute_modal_wind_direction(distribution)
        assert xarray_snapshot == result


class TestWeightedCompositeSnapshot:
    """Snapshot tests for weighted composite burn probability."""

    def test_create_weighted_composite_basic(self, xarray_snapshot):
        """Snapshot test for basic weighted composite."""
        ny, nx = 10, 10
        lat = np.linspace(35, 36, ny)
        lon = np.linspace(-120, -119, nx)

        # Create directional burn probability layers with spatial variation
        direction_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'circular']
        data_vars = {}
        for i, lab in enumerate(direction_labels):
            # Each direction has different spatial pattern
            y_grid, x_grid = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
            pattern = 0.01 + 0.001 * i * np.sin(y_grid / 2) * np.cos(x_grid / 2)
            data_vars[lab] = (('latitude', 'longitude'), pattern.astype(np.float32))

        bp = xr.Dataset(data_vars, coords={'latitude': lat, 'longitude': lon})

        # Create spatially varying distribution
        np.random.seed(42)
        probs = np.random.dirichlet([1] * 8, size=(ny, nx))  # Sum to 1 per pixel
        dist = xr.DataArray(
            probs.transpose(2, 0, 1).astype(np.float32),
            dims=['wind_direction', 'latitude', 'longitude'],
            coords={'wind_direction': direction_labels[:8], 'latitude': lat, 'longitude': lon},
            name='wind_direction_distribution',
        )

        result = create_weighted_composite_bp_map(bp, dist).to_dataset()
        assert xarray_snapshot == result

    def test_create_weighted_composite_with_missing_data(self, xarray_snapshot):
        """Snapshot test for weighted composite with missing fire weather data."""
        ny, nx = 8, 8
        lat = np.linspace(35, 36, ny)
        lon = np.linspace(-120, -119, nx)

        direction_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'circular']
        data_vars = {
            lab: (('latitude', 'longitude'), np.full((ny, nx), 0.01 * (i + 1), dtype=np.float32))
            for i, lab in enumerate(direction_labels)
        }
        bp = xr.Dataset(data_vars, coords={'latitude': lat, 'longitude': lon})

        # Create distribution with some pixels having no fire weather (all zeros)
        probs = np.zeros((8, ny, nx), dtype=np.float32)
        probs[:, :4, :] = 0.125  # Top half has uniform distribution
        # Bottom half has all zeros (no fire weather)

        dist = xr.DataArray(
            probs,
            dims=['wind_direction', 'latitude', 'longitude'],
            coords={'wind_direction': direction_labels[:8], 'latitude': lat, 'longitude': lon},
            name='wind_direction_distribution',
        )

        result = create_weighted_composite_bp_map(bp, dist).to_dataset()
        assert xarray_snapshot == result


class TestFosbergIndexSnapshot:
    """Snapshot tests for Fosberg Fire Weather Index."""

    def test_fosberg_fire_weather_index(self, xarray_snapshot):
        """Snapshot test for FFWI calculation."""
        ny, nx = 10, 10
        lat = np.linspace(35, 36, ny)
        lon = np.linspace(-120, -119, nx)

        # Create sample meteorological data with realistic ranges
        np.random.seed(42)
        hurs = xr.DataArray(
            np.random.uniform(10, 90, (ny, nx)).astype(np.float32),
            dims=['latitude', 'longitude'],
            coords={'latitude': lat, 'longitude': lon},
            attrs={'units': 'percent'},
        )

        T2 = xr.DataArray(
            np.random.uniform(273.15 + 10, 273.15 + 35, (ny, nx)).astype(np.float32),
            dims=['latitude', 'longitude'],
            coords={'latitude': lat, 'longitude': lon},
            attrs={'units': 'K'},
        )

        sfcWind = xr.DataArray(
            np.random.uniform(0, 15, (ny, nx)).astype(np.float32),
            dims=['latitude', 'longitude'],
            coords={'latitude': lat, 'longitude': lon},
            attrs={'units': 'm/s'},
        )

        result = fosberg_fire_weather_index(hurs, T2, sfcWind).to_dataset()
        assert xarray_snapshot == result


class TestWindInformedBurnProbabilitySnapshot:
    """Snapshot tests for wind-informed burn probability creation."""

    @pytest.mark.parametrize(
        'region_id',
        [
            pytest.param('california-coast', id='california-coast'),
            pytest.param('colorado-rockies', id='colorado-rockies'),
            pytest.param('seattle-area', id='seattle-area'),
            pytest.param('georgia-piedmont', id='georgia-piedmont'),
            pytest.param('arizona-desert', id='arizona-desert'),
        ],
    )
    def test_create_wind_informed_burn_probability_regions(
        self, xarray_snapshot, get_wind_informed_burn_probability, region_id
    ):
        """Snapshot test for wind-informed burn probability creation on different regions.

        This test captures the output of the wind-informed burn probability calculation,
        which combines directional convolution with Riley burn probability data and
        wind direction distributions. Tests multiple geographic regions to ensure
        correct behavior across different landscapes and wind patterns.

        Note: Uses cached calculations (session-scoped) to share data with other tests.
        """
        result = get_wind_informed_burn_probability(region_id)

        # Snapshot the result
        assert xarray_snapshot == result


class TestWindAdjustedRiskSnapshot:
    """Snapshot tests for the complete wind-adjusted risk calculation pipeline."""

    @pytest.mark.parametrize(
        'region_id',
        [
            pytest.param('california-coast', id='california-coast'),
            pytest.param('colorado-rockies', id='colorado-rockies'),
            pytest.param('seattle-area', id='seattle-area'),
            pytest.param('georgia-piedmont', id='georgia-piedmont'),
            pytest.param('arizona-desert', id='arizona-desert'),
        ],
    )
    def test_calculate_wind_adjusted_risk_regions(
        self, xarray_snapshot, get_wind_adjusted_risk, region_id
    ):
        """Snapshot test for wind-adjusted risk calculation on different regions.

        This test captures the full output of the main risk calculation pipeline,
        including all intermediate datasets and transformations. Tests multiple
        geographic regions to ensure the pipeline works correctly across different
        landscapes and conditions.

        Note: Uses cached risk calculations (session-scoped) shared with pipeline tests.
        """
        # Get cached risk calculation (shared with pipeline tests)
        result = get_wind_adjusted_risk(region_id)

        # Snapshot the full result
        assert xarray_snapshot == result
