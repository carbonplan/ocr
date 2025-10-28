import typing
import warnings

import numpy as np
import xarray as xr
from odc.geo.xr import assign_crs, xr_reproject
from scipy.ndimage import rotate

from ocr import catalog
from ocr.utils import geo_sel

CARDINAL_AND_ORDINAL = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def generate_weights(
    method: typing.Literal['skewed', 'circular_focal_mean'] = 'skewed',
    kernel_size: float = 81.0,
    circle_diameter: float = 35.0,
) -> np.ndarray:
    """Generate a 2D array of weights for a circular kernel.

    Parameters
    ----------
    method : str, optional
        The method to use for generating weights. Options are 'skewed' or 'circular_focal_mean'.
        'skewed' generates an elliptical kernel to simulate wind directionality.
        'circular_focal_mean' generates a circular kernel, by default 'skewed'
    kernel_size : float, optional
        The size of the kernel, by default 81.0
    circle_diameter : float, optional
        The diameter of the circle, by default 35.0

    Returns
    -------
    weights : np.ndarray
        A 2D array of weights for the circular kernel.
    """
    if method == 'circular_focal_mean':
        x, y = np.meshgrid(
            np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1),
            np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1),
        )
        distances = np.sqrt(x**2 + y**2)
        inside = distances <= circle_diameter // 2 + 1
        weights = inside / inside.sum()

    elif method == 'skewed':
        # elliptical kernel
        a = 4  # semi-major axis
        b = 2  # semi-minor axis
        x = np.linspace(-kernel_size // 2, kernel_size // 2 + 1, int(kernel_size))
        y = np.linspace(-kernel_size // 2, kernel_size // 2 + 1, int(kernel_size))
        xx, yy = np.meshgrid(x, y)

        # Ellipse equation
        ellipse = ((xx / a) ** 2 + (yy / b) ** 2) <= 10

        weights = np.roll(ellipse, -5)
        # Normalize to sum to 1.0 if there are any non-zero entries
        weights = weights.astype(np.float32)
        s = float(weights.sum())
        if s > 0:
            weights = weights / s

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
    kernels : dict[str, np.ndarray]
        A dictionary of 2D arrays of weights for circular kernels oriented in different directions.
    """
    weights_dict = {}
    rotating_angles = np.arange(0, 360, 45)
    wind_direction_labels = ['W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW']
    for angle, direction in zip(rotating_angles, wind_direction_labels):
        base = generate_weights(
            method='skewed', kernel_size=kernel_size, circle_diameter=circle_diameter
        ).astype(np.float32)
        rotated = rotate(
            base,
            angle=angle,
            reshape=False,  # keep original shape
            order=1,  # bilinear to reduce ringing
            mode='nearest',
            prefilter=False,
        )

        if angle in [45, 135, 225, 315]:
            # TODO, @orianac, i presume this cropping only applies to kernel_size=81.0, circle_diameter=35.0. If so, what should the cropping be for other kernel sizes and circle diameters?
            rotated = rotated[17:98, 17:98]

        # Remove tiny negative interpolation artifacts, renormalize
        rotated = np.clip(rotated, 0.0, None)
        weights_dict[direction] = rotated

    circ = generate_weights(
        method='circular_focal_mean',
        kernel_size=kernel_size,
        circle_diameter=circle_diameter,
    ).astype(np.float32)
    circ = np.clip(circ, 0, None)

    weights_dict['circular'] = circ

    # re-normalize all weights to ensure sum equals 1.0
    for direction in weights_dict:
        s = weights_dict[direction].sum()
        if s > 0:
            weights_dict[direction] = weights_dict[direction] / s
    return weights_dict


def apply_wind_directional_convolution(
    da: xr.DataArray,
    iterations: int = 3,
    kernel_size: float = 81.0,
    circle_diameter: float = 35.0,
) -> xr.Dataset:
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
    ds : xr.Dataset
        The Dataset with the directional convolution applied
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
        attrs=da.attrs,
    )
    # Preserve any CRS-related coordinates
    for coord in ['spatial_ref', 'crs']:
        if coord in da.coords:
            spread_results = spread_results.assign_coords({coord: da.coords[coord]})

    for direction, weights in weights_dict.items():
        arr = spread_results[direction].values
        for _ in range(iterations):
            valid_mask = (arr > 0).astype(np.float32)
            convolved_mask = cv.filter2D(valid_mask, ddepth=-1, kernel=weights)
            # why does `filter2D` produce tiny negatives and tiny positives? we shouldn't have to deal with this
            # if we do clip we should ensure we're only clipping VERY small numbers (e.g. e-12)
            convolved_mask = np.where(convolved_mask < 10e-10, 0.0, convolved_mask)

            convolved_arr = cv.filter2D(arr, ddepth=-1, kernel=weights)
            convolved_arr = np.where(convolved_arr < 10e-10, 0.0, convolved_arr)
            arr = np.where(convolved_mask > 0, convolved_arr / convolved_mask, 0.0)
        # clip residual tiny negatives
        arr = np.where(arr < 0, 0.0, arr)
        spread_results[direction] = (da.dims, arr.astype(np.float32))
    return spread_results


def classify_wind_directions(wind_direction_ds: xr.DataArray) -> xr.DataArray:
    """
    Classify wind directions into 8 cardinal directions (0-7).
    The classification is:

    0: North (337.5-22.5)
    1: Northeast (22.5-67.5)
    2: East (67.5-112.5)
    3: Southeast (112.5-157.5)
    4: South (157.5-202.5)
    5: Southwest (202.5-247.5)
    6: West (247.5-292.5)
    7: Northwest (292.5-337.5)

    Parameters
    ----------
    wind_direction_ds : xarray.DataArray
        DataArray containing wind direction in degrees (0-360)

    Returns
    -------
    result : xarray.DataArray
        DataArray with wind directions classified as integers 0-7
    """
    # todo - make tests that ensure that our orientation is always initialized
    # to the north (0 index = north like direction_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'circular']
    #  and that it's centered (i.e. North is -22.5 to 22.5)
    # bins for classification (degrees)
    bins = np.arange(22.5, 360, 45)

    def classify_block(block):
        # Handle the North wind edge case (337.5-360 and 0-22.5)
        # We use modulo 360 to ensure all values are in 0-360 range
        normalized_directions = block % 360

        # For values between 337.5 and 360, we want to classify them as North (0)
        north_mask = normalized_directions >= 337.5

        classification = np.digitize(normalized_directions, bins=bins)

        # explicitly set the North classification for values >= 337.5
        classification[north_mask] = 0

        # preserve NaN values instead of replacing them
        classification = np.where(np.isnan(block), np.nan, classification)

        return classification.astype(np.float32)

    result = wind_direction_ds.copy()

    if hasattr(wind_direction_ds.data, 'map_blocks'):
        # Apply the function using map_blocks
        result.data = wind_direction_ds.data.map_blocks(classify_block, dtype=np.float32)
    else:
        # Fall back for non-dask arrays
        result.data = classify_block(wind_direction_ds.data)

    result.attrs = {}
    # rename variable to wind_direction_classification
    result = result.rename('wind_direction_classification')
    result.attrs['long_name'] = 'wind direction classified into 8 cardinal directions (0-7)'
    result.attrs['short_name'] = 'wind_direction_classification'
    result.attrs['direction_labels'] = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    return result


def _bp_dataset_to_direction_da(bp: xr.Dataset) -> xr.DataArray:
    """
    Convert a Dataset with variables named after the 8 directions + 'circular'
    into a single DataArray with a 'direction' dimension.
    """

    missing = [d for d in CARDINAL_AND_ORDINAL if d not in bp]
    if missing:
        raise KeyError(f'bp dataset is missing expected direction variable(s): {missing}')
    bp_da = (
        bp[CARDINAL_AND_ORDINAL]
        .to_array(dim='direction')
        .assign_coords(direction=CARDINAL_AND_ORDINAL)
    )
    return bp_da


def create_weighted_composite_bp_map(
    bp: xr.Dataset,
    wind_direction_distribution: xr.DataArray,
    *,
    distribution_direction_dim: str = 'wind_direction',
    weight_sum_tolerance: float = 1e-5,
) -> xr.DataArray:
    """Create a weighted composite burn probability map using wind direction distribution.

    Parameters
    ----------
    bp : xr.Dataset
        Dataset containing 9 directional burn probability layers with variables
        named ['N','NE','E','SE','S','SW','W','NW','circular'] produced by
        `apply_wind_directional_convolution`.
    wind_direction_distribution : xr.DataArray
        Probability distribution over 8 cardinal directions with dimension
        'wind_direction' and length 8, matching direction labels:
        ['N','NE','E','SE','S','SW','W','NW'] (order must align). Values should
        sum to 1 where fire-weather hours exist; may be all 0 where none exist.
    distribution_direction_dim : str, optional
        Name of the dimension in `wind_direction_distribution` that holds the
        direction labels, by default 'wind_direction'.
    weight_sum_tolerance : float, optional
        Tolerance for deviation from 1.0 in the sum of weights, by default


    Returns
    -------
    weighted : xr.DataArray
        Weighted composite burn probability with same spatial dims as inputs.
        Name: 'wind_weighted_bp'. Missing (all-zero) distributions yield NaN.
    """

    bp_da = _bp_dataset_to_direction_da(bp)

    # Prepare distribution: ensure a 'direction' coord with labels matching CARDINAL_8
    if distribution_direction_dim != 'direction':
        wind_direction_distribution = wind_direction_distribution.rename(
            {distribution_direction_dim: 'direction'}
        )
    if 'direction' not in wind_direction_distribution.dims:
        raise ValueError('Distribution must have a direction dimension after renaming.')

    # Attach labels if they are not present (numeric 0..7 assumed in correct order)
    if (
        'direction' not in wind_direction_distribution.coords
        or len(wind_direction_distribution['direction']) != 8
    ):
        wind_direction_distribution = wind_direction_distribution.assign_coords(
            direction=CARDINAL_AND_ORDINAL
        )

    # Sanity checks
    if set(wind_direction_distribution['direction'].values) != set(CARDINAL_AND_ORDINAL):
        raise ValueError(
            f'Distribution direction labels must match {CARDINAL_AND_ORDINAL}; got {list(wind_direction_distribution["direction"].values)}'
        )

    # Identify invalid / missing rows
    any_nan_row = wind_direction_distribution.isnull().any(dim='direction')
    negative_weights = (wind_direction_distribution < 0).any()
    if bool(negative_weights):
        raise ValueError('Distribution contains negative probabilities.')

    row_sum = wind_direction_distribution.sum(dim='direction')

    has_fire_weather = row_sum > 0
    valid_rows = has_fire_weather & (~any_nan_row)

    normalized_distribution = xr.where(
        valid_rows,
        wind_direction_distribution / row_sum.where(valid_rows),
        wind_direction_distribution,
    )

    post_sum = normalized_distribution.sum(dim='direction').where(valid_rows)
    off = (np.abs(post_sum - 1) > weight_sum_tolerance).sum()
    if int(off) > 0:
        warnings.warn(
            f'{int(off)} valid pixel(s) have weight sums outside tolerance {weight_sum_tolerance}.',
            UserWarning,
            stacklevel=2,
        )

    cardinal_and_ordinal_bp = bp_da.sel(direction=CARDINAL_AND_ORDINAL)
    weighted = (cardinal_and_ordinal_bp * normalized_distribution).sum(dim='direction')

    weighted = weighted.where(valid_rows)

    weighted.name = 'wind_weighted_bp'
    weighted.attrs.update(
        {
            'composition': 'weighted',
            'long_name': 'Weighted composite BP map using wind direction distribution',
            'direction_labels': CARDINAL_AND_ORDINAL,
            'weights_source': wind_direction_distribution.name or 'wind_direction_distribution',
            'note': 'Rows with no fire-weather hours or NaNs are NaN',
        }
    )

    return weighted


def create_wind_informed_burn_probability(
    wind_direction_distribution_30m_4326: xr.DataArray,
    riley_270m_5070: xr.Dataset,
) -> xr.DataArray:
    """Create wind-informed burn probability dataset by applying directional convolution and creating a weighted composite burn probability map.

    Parameters
    ----------
    wind_direction_distribution_30m_4326 : xr.DataArray
        Wind direction distribution data at 30m resolution in EPSG:4326 projection.
    riley_270m_5070 : xr.DataArray
        Riley et al. (2011) burn probability data at 270m resolution in EPSG:5070 projection.

    Returns
    -------
    smoothed_final_bp : xr.DataArray
        Smoothed wind-informed burn probability data at 30m resolution in EPSG:4326 projection.
    """
    import cv2 as cv

    ## gap fill riley 270 only filling NaN pixels which are surrounded by valid data on four sides
    valid_pixels = riley_270m_5070['BP'] > 0
    # shifting will introduce NaNs! So we need to clip along boundaries
    surrounded_by_four_valid_values = (
        valid_pixels.shift(x=1)
        + valid_pixels.shift(x=-1)
        + valid_pixels.shift(y=1)
        + valid_pixels.shift(y=-1)
    ) == 4
    # 270m 5070 projection mask of every pixel which is surrounded by valid data on four sides
    nans_surrounded_by_four_valid_values = xr.where(
        (valid_pixels == 0) & surrounded_by_four_valid_values, 1, 0
    )
    # where nans_surrounded_by_four_valid_values is true, fill with a 3x3 moving window average of valid pixels
    rolling_mean = riley_270m_5070.rolling({'x': 3, 'y': 3}, center=True, min_periods=1).mean(
        skipna=True
    )
    gap_filled_riley_2011_270m_5070_subset = xr.where(
        nans_surrounded_by_four_valid_values, rolling_mean, riley_270m_5070
    )

    wind_direction_distribution_30m_4326 = assign_crs(
        wind_direction_distribution_30m_4326, 'EPSG:4326'
    )
    gap_filled_riley_2011_270m_5070_subset = assign_crs(
        gap_filled_riley_2011_270m_5070_subset, 'EPSG:5070'
    )
    ## reproject to the 30m 4326 projection (use riley_30m_4326 as variable name)
    target_geobox = wind_direction_distribution_30m_4326.odc.geobox
    riley_30m_4326 = xr_reproject(
        gap_filled_riley_2011_270m_5070_subset,
        how=target_geobox,
        resampling='nearest',
    )
    riley_30m_4326 = riley_30m_4326.assign_coords(
        {
            'latitude': wind_direction_distribution_30m_4326.latitude,
            'longitude': wind_direction_distribution_30m_4326.longitude,
        }
    )

    blurred_bp_30m_4326 = apply_wind_directional_convolution(riley_30m_4326['BP'], iterations=3)
    wind_informed_bp_30m_4326 = create_weighted_composite_bp_map(
        blurred_bp_30m_4326, wind_direction_distribution_30m_4326
    )
    wind_informed_bp_30m_4326 = assign_crs(wind_informed_bp_30m_4326, 'EPSG:4326')

    wind_informed_bp_30m_4326 = wind_informed_bp_30m_4326.assign_coords(
        {
            'latitude': wind_direction_distribution_30m_4326.latitude,
            'longitude': wind_direction_distribution_30m_4326.longitude,
        }
    )
    # gap fill any zeroes remaining in riley using the wind-smeared numbers

    # retain original Riley et al. (2025) burn probability where there are valid numbers at a 270m scale (but based upon
    # the dataset reprojected and interpolated to a 30m EPSG:4326 grid). anywhere there are no valid numbers use the
    # wind smeared values
    wind_informed_bp_combined = xr.where(
        riley_30m_4326['BP'] == 0, wind_informed_bp_30m_4326, riley_30m_4326['BP']
    )

    # smooth using a 21x21 Gaussian filter
    smoothed_bp = xr.apply_ufunc(
        cv.GaussianBlur,
        wind_informed_bp_combined.chunk(latitude=-1, longitude=-1),
        input_core_dims=[['latitude', 'longitude']],
        output_core_dims=[['latitude', 'longitude']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32],
        kwargs={'ksize': (21, 21), 'sigmaX': 0},
    )
    smoothed_bp.name = 'BP'

    smoothed_bp.attrs = {
        'long_name': 'Wind-informed Burn Probability',
        'description': 'Wind-informed Burn Probability created by applying directional convolution and weighted composite using wind direction distribution',
    }
    return smoothed_bp


def calculate_wind_adjusted_risk(
    *,
    x_slice: slice,
    y_slice: slice,
    buffer: float = 0.15,
) -> xr.Dataset:
    """Calculate wind-adjusted fire risk using climate run and wildfire risk datasets.

    Parameters
    ----------
    x_slice : slice
        Slice object for selecting longitude range.
    y_slice : slice
        Slice object for selecting latitude range.
    buffer : float, optional
        Buffer size in degrees to add around the region for edge effect handling (default 0.15).
        For 30m EPSG:4326 data, 0.15 degrees ≈ 16.7 km ≈ 540 pixels.
        This buffer ensures neighborhood operations (convolution, Gaussian smoothing) have
        adequate context at boundaries.

    Returns
    -------
    fire_risk : xr.Dataset
        Dataset containing wind-adjusted fire risk variables.
    """

    buffered_x_slice = slice(x_slice.start - buffer, x_slice.stop + buffer, x_slice.step)
    buffered_y_slice = slice(y_slice.start - buffer, y_slice.stop + buffer, y_slice.step)

    riley_2011_30m_4326 = catalog.get_dataset('riley-et-al-2025-2011-30m-4326').to_xarray()[['BP']]
    riley_2047_30m_4326 = catalog.get_dataset('riley-et-al-2025-2047-30m-4326').to_xarray()[['BP']]
    riley_2011_270m_5070 = catalog.get_dataset('riley-et-al-2025-2011-270m-5070').to_xarray()[
        ['BP', 'spatial_ref']
    ]
    riley_2011_270m_5070 = assign_crs(riley_2011_270m_5070, 'EPSG:5070')
    riley_2047_270m_5070 = catalog.get_dataset('riley-et-al-2025-2047-270m-5070').to_xarray()[
        ['BP', 'spatial_ref']
    ]
    riley_2047_270m_5070 = assign_crs(riley_2047_270m_5070, 'EPSG:5070')

    rps_30 = catalog.get_dataset('scott-et-al-2024-30m-4326').to_xarray()[['BP', 'CRPS', 'RPS']]

    riley_2011_30m_4326_subset = riley_2011_30m_4326.sel(
        latitude=buffered_y_slice, longitude=buffered_x_slice
    )
    riley_2047_30m_4326_subset = riley_2047_30m_4326.sel(
        latitude=buffered_y_slice, longitude=buffered_x_slice
    )

    # west, south, east, north = bbox

    bbox = (
        buffered_x_slice.start,
        buffered_y_slice.start,
        buffered_x_slice.stop,
        buffered_y_slice.stop,
    )
    riley_2011_270m_5070_subset = geo_sel(
        riley_2011_270m_5070,
        bbox=bbox,
        crs_wkt=riley_2011_270m_5070.spatial_ref.attrs['crs_wkt'],
    )
    riley_2047_270m_5070_subset = geo_sel(
        riley_2047_270m_5070,
        bbox=bbox,
        crs_wkt=riley_2047_270m_5070.spatial_ref.attrs['crs_wkt'],
    )

    wind_direction_distribution_30m_4326 = (
        catalog.get_dataset('conus404-ffwi-p99-wind-direction-distribution-30m-4326')
        .to_xarray()
        .wind_direction_distribution.sel(latitude=buffered_y_slice, longitude=buffered_x_slice)
        .load()
    )

    wind_informed_bp_combined_2011 = create_wind_informed_burn_probability(
        wind_direction_distribution_30m_4326=wind_direction_distribution_30m_4326,
        riley_270m_5070=riley_2011_270m_5070_subset,
    )

    wind_informed_bp_combined_2047 = create_wind_informed_burn_probability(
        wind_direction_distribution_30m_4326=wind_direction_distribution_30m_4326,
        riley_270m_5070=riley_2047_270m_5070_subset,
    )

    # clip to original x_slice, y_slice
    wind_informed_bp_combined_2011 = wind_informed_bp_combined_2011.sel(
        latitude=y_slice, longitude=x_slice
    )
    wind_informed_bp_combined_2047 = wind_informed_bp_combined_2047.sel(
        latitude=y_slice, longitude=x_slice
    )
    riley_2011_30m_4326_subset = riley_2011_30m_4326_subset.sel(latitude=y_slice, longitude=x_slice)
    riley_2047_30m_4326_subset = riley_2047_30m_4326_subset.sel(latitude=y_slice, longitude=x_slice)

    fire_risk = xr.Dataset()

    # Note: for QA. Remove in further production versions
    rps_30_subset = rps_30.sel(latitude=y_slice, longitude=x_slice)
    fire_risk['USFS_RPS'] = rps_30_subset['RPS']

    # wind_risk_2011 (our wind-informed RPS value)
    fire_risk['wind_risk_2011'] = wind_informed_bp_combined_2011 * rps_30_subset['CRPS']
    fire_risk['wind_risk_2011'].attrs['description'] = (
        'Wind-informed RPS for 2011 calculated as wind-informed BP * CRPS'
    )
    # wind_risk_2047 (our wind-informed RPS value)
    fire_risk['wind_risk_2047'] = wind_informed_bp_combined_2047 * rps_30_subset['CRPS']
    fire_risk['wind_risk_2047'].attrs['description'] = (
        'Wind-informed RPS for 2047 calculated as wind-informed BP * CRPS'
    )

    # burn_probability_2011 (our wind-informed BP value)
    fire_risk['burn_probability_2011'] = wind_informed_bp_combined_2011
    fire_risk['burn_probability_2011'].attrs['description'] = (
        'Wind-informed burn probability for 2011 calculated as wind-informed BP'
    )

    # burn_probability_2047 (our wind-informed BP value)
    fire_risk['burn_probability_2047'] = wind_informed_bp_combined_2047
    fire_risk['burn_probability_2047'].attrs['description'] = (
        'Wind-informed burn probability for 2047 calculated as wind-informed BP'
    )

    # conditional_risk (from USFS Scott 2024)(RDS-2020-0016-2)
    fire_risk['conditional_risk_usfs'] = rps_30_subset['CRPS']

    # burn_probability_usfs_2011 (BP from Riley 2025 (RDS-2025-0006))
    fire_risk['burn_probability_usfs_2011'] = riley_2011_30m_4326_subset['BP']

    # burn_probability_usfs_2047 (BP from Riley 2025 (RDS-2025-0006))
    fire_risk['burn_probability_usfs_2047'] = riley_2047_30m_4326_subset['BP']

    return fire_risk.drop_vars(['spatial_ref', 'crs', 'quantile'], errors='ignore')


def direction_histogram(data_array: xr.DataArray) -> xr.DataArray:
    """
    Compute direction histogram on xarray DataArray with dask chunks.

    Parameters
    -----------
    data_array : xarray.DataArray
        Input data array containing direction indices (expected to be integers 0-7)

    Returns
    -------
    xarray.DataArray
        Normalized histogram counts as a probability distribution
    """

    def _compute_bin_count(arr):
        # Filter out negative values
        valid_arr = arr[arr >= 0]

        # Return immediately if no valid data
        if len(valid_arr) == 0:
            return np.zeros(8, dtype=np.float32)

        int_arr = valid_arr.astype(np.int64)
        counts = np.bincount(int_arr, minlength=8)

        total = counts.sum()

        # Avoid division if possible
        return counts / total

    fraction = xr.apply_ufunc(
        _compute_bin_count,
        data_array,
        input_core_dims=[['time']],
        output_core_dims=[['wind_direction']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32],
        kwargs={},
        dask_gufunc_kwargs={
            'output_sizes': {'wind_direction': 8},
        },
        keep_attrs=False,
    )
    fraction = fraction.rename('wind_direction_histogram')
    fraction.attrs = {
        'long_name': 'Wind Direction Histogram',
        'units': 'probability',
    }
    return fraction


def fosberg_fire_weather_index(
    hurs: xr.DataArray, T2: xr.DataArray, sfcWind: xr.DataArray
) -> xr.DataArray:
    """
    Calculate the Fosberg Fire Weather Index (FFWI) based on relative humidity, temperature, and wind speed.
    taken from https://wikifire.wsl.ch/tiki-indexb1d5.html?page=Fosberg+fire+weather+index&structure=Fire
    hurs, T2, sfcWind are arrays

    Parameters
    ----------
    hurs : xr.DataArray
        Relative humidity in percentage (0-100).
    T2 : xr.DataArray
        Temperature
    sfcWind : xr.DataArray
        Wind speed in meters per second.


    Returns
    -------
    xr.DataArray
        Fosberg Fire Weather Index (FFWI).
    """
    import pint_xarray  # noqa: F401

    # Convert temperature to Fahrenheit
    T2 = T2.pint.quantify().pint.to('degF').pint.dequantify()

    # Convert wind speed to meters per second if necessary
    sfcWind = sfcWind.pint.quantify().pint.to('m/s').pint.dequantify()

    hurs = hurs.pint.quantify().pint.to('percent').pint.dequantify()

    emc = xr.where(
        (hurs >= 0) & (hurs < 10),
        0.03229 + 0.281073 * hurs - 0.000578 * hurs * T2,
        xr.where(
            (hurs >= 10) & (hurs < 50),
            2.22749 + 0.160107 * hurs - 0.01478 * T2,
            xr.where(
                (hurs >= 50) & (hurs <= 100),
                21.0606 + 0.005565 * hurs**2 - 0.00035 * hurs * T2 - 0.483199 * hurs,
                np.nan,
            ),
        ),
    )

    # emc: equilibrium moisture content, units are %
    nu = 1 - 2 * (emc / 30) + 1.5 * ((emc / 30) ** 2) - 0.5 * ((emc / 30) ** 3)
    ffwi = nu * np.sqrt(1 + (sfcWind**2))
    ffwi = typing.cast(xr.DataArray, ffwi)
    ffwi.name = 'FFWI'
    ffwi.attrs['long_name'] = 'Fosberg Fire Weather Index'
    ffwi.attrs['units'] = 'dimensionless'

    return ffwi


def compute_wind_direction_distribution(
    direction: xr.DataArray, fire_weather_mask: xr.DataArray
) -> xr.Dataset:
    """
    Compute the wind direction distribution during fire weather conditions.

    Parameters
    ----------
    direction : xr.DataArray
        Wind direction in degrees (0-360).
    fire_weather_mask : xr.DataArray
        Boolean mask indicating fire weather conditions.

    Returns
    -------
    wind_direction_hist : xr.Dataset
        Wind direction histogram during fire weather conditions.
    """
    # Classify wind directions into 8 cardinal directions
    classified_directions = classify_wind_directions(direction)

    # Apply fire weather mask to filter relevant data
    masked_directions = classified_directions.where(fire_weather_mask)

    # Compute histogram of wind directions during fire weather conditions
    wind_direction_hist = direction_histogram(masked_directions)

    wind_direction_hist.name = 'wind_direction_distribution'
    wind_direction_hist.attrs['long_name'] = 'Wind direction distribution during fire-weather hours'
    wind_direction_hist.attrs['description'] = (
        'Fraction of hours in each of 8 cardinal and ordinal directions during hours meeting fire weather criteria'
    )

    hist_dim = 'wind_direction'
    chunk_dict = {dim: -1 for dim in wind_direction_hist.dims if dim != hist_dim}

    return wind_direction_hist.to_dataset().chunk(chunk_dict)


def compute_modal_wind_direction(distribution: xr.DataArray) -> xr.Dataset:
    """
    Compute the modal wind direction from the wind direction distribution.

    Parameters
    ----------
    distribution : xr.DataArray
        Wind direction distribution.

    Returns
    -------
    mode : xr.Dataset
        Modal wind direction.
    """
    # Identify pixels with any fire-weather hours (probabilities sum to 1 else 0)
    any_fire_weather = distribution.sum(dim='wind_direction') > 0

    # TODO: Handling ties.
    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html mentions that in case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
    # Defensive: ensure we actually have a DataArray (helpful for static type checkers)
    assert isinstance(distribution, xr.DataArray)
    dist_da: xr.DataArray = distribution  # explicit alias for type checkers
    idx = dist_da.argmax(dim='wind_direction')
    # Cast for static type checkers that may not follow xarray's dynamic return
    from typing import cast as _cast

    idx_da = _cast(xr.DataArray, idx)
    mode = idx_da.where(any_fire_weather)  # type: ignore[attr-defined]
    if {'x', 'y'}.issubset(mode.dims):
        mode = mode.chunk({'x': -1, 'y': -1})
    mode.name = 'wind_direction_mode'
    mode.attrs.update(
        {
            'long_name': 'Modal wind direction during fire-weather hours',
            'description': 'Most frequent of 8 cardinal directions during hours meeting fire weather criteria',
            'direction_labels': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
        }
    )
    return mode.to_dataset()
