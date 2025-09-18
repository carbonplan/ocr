import typing
import warnings

import numpy as np
import xarray as xr

from ocr import catalog


def generate_weights(
    method: typing.Literal['skewed', 'circular_focal_mean'] = 'skewed',
    kernel_size: float = 81.0,
    circle_diameter: float = 35.0,
) -> np.ndarray:
    """Generate a 2D array of weights for a circular kernel."""
    if method == 'circular_focal_mean':
        x, y = np.meshgrid(
            np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1),
            np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1),
        )
        distances = np.sqrt(x**2 + y**2)
        inside = distances <= circle_diameter // 2 + 1
        weights = inside / inside.sum()

    elif method == 'skewed':
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
        arr = spread_results[direction].values
        for _ in range(iterations):
            arr = cv.filter2D(arr, ddepth=-1, kernel=weights)
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


def compute_mode_along_time(direction_indices_ds: xr.DataArray) -> xr.DataArray:
    """
    Compute the most common wind direction at each location over time.
    Uses map_blocks instead of apply_ufunc for better performance.

    Parameters:
    -----------
    direction_indices_ds : xarray.DataArray
        DataArray with dimensions (time, latitude, longitude) containing wind direction indices

    Returns:
    --------
    xarray.DataArray
        DataArray with dimensions (latitude, longitude) containing the most common wind direction
    """

    # Define the function to compute mode on each block
    def _compute_mode_block(block):
        # For each lat/lon point, compute mode along time dimension
        result = np.zeros(block.shape[1:], dtype=np.float32)

        # Iterate over each lat/lon point
        # We could vectorize this, but explicit iteration makes the axis handling clearer
        for i in np.ndindex(block.shape[1:]):
            # Extract the time series for this lat/lon point
            time_series = block[:, *i]

            # Remove NaNs and placeholders
            valid_values = time_series[~np.isnan(time_series)]

            if len(valid_values) == 0:
                result[i] = np.nan  # Return NaN if no valid values
            else:
                # Find the mode
                values, counts = np.unique(valid_values, return_counts=True)
                result[i] = values[np.argmax(counts)]

        return result

    if hasattr(direction_indices_ds.data, 'map_blocks'):
        # Use map_blocks to apply our function
        # The input array shape is (time, lat, lon)
        # We want to drop the time dimension, so output will be (lat, lon)
        result_data = direction_indices_ds.data.map_blocks(
            _compute_mode_block,
            dtype=np.float32,
            drop_axis=0,  # Drop the time dimension (axis 0)
            chunks=direction_indices_ds.data.chunks[1:],  # Output chunks match lat/lon chunks
        )
    else:
        # Fall back for non-dask arrays
        result_data = _compute_mode_block(direction_indices_ds.data)

    result = xr.DataArray(
        result_data,
        dims=direction_indices_ds.dims[1:],  # Use lat/lon dimensions
        coords={
            dim: direction_indices_ds[dim] for dim in direction_indices_ds.dims[1:]
        },  # Copy coordinates
        name='modal_wind_direction',
    )

    # Copy attributes from the original DataArray
    if hasattr(direction_indices_ds, 'attrs'):
        result.attrs = direction_indices_ds.attrs.copy()

    result.attrs.update(
        {
            'long_name': 'Most common wind direction',
            'description': 'Modal wind direction over the time period',
        }
    )

    return result


def create_composite_bp_map(bp: xr.Dataset, wind_directions: xr.DataArray) -> xr.DataArray:
    direction_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'circular']
    # reorder the differently blurred BP and then turn it into a DataArray and assign the coords
    bp_da = bp[direction_labels].to_array(dim='direction').assign_coords(direction=direction_labels)
    # select the entry that corresponds to the wind direction index
    # Preserve NaNs & truly invalid indices ( <0 or >8 ) as missing.
    # Indices 0-7 correspond to cardinal directions; 8 is the explicit 'circular' direction
    missing = wind_directions.isnull() | (wind_directions < 0) | (wind_directions > 8)
    # if missing.sum() > 0. let's warn
    if missing.sum() > 0:
        warnings.warn(
            f'Missing and/or invalid wind direction data for {missing.sum().data!r} points.',
            UserWarning,
            stacklevel=2,
        )
    # For valid positions (including 8) we can index directly
    safe_indexer = wind_directions.where(~missing, 0).astype(np.int16)
    out = bp_da.isel(direction=safe_indexer).where(~missing)
    return out


CARDINAL_AND_ORDINAL = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
ALL_9 = CARDINAL_AND_ORDINAL + ['circular']


def _bp_dataset_to_direction_da(bp: xr.Dataset) -> xr.DataArray:
    """
    Convert a Dataset with variables named after the 8 directions + 'circular'
    into a single DataArray with a 'direction' dimension.
    """

    missing = [d for d in ALL_9 if d not in bp]
    if missing:
        raise KeyError(f'bp dataset is missing expected direction variable(s): {missing}')
    bp_da = bp[ALL_9].to_array(dim='direction').assign_coords(direction=ALL_9)
    return bp_da


def create_weighted_composite_bp_map(
    bp: xr.Dataset,
    distribution: xr.DataArray,
    *,
    distribution_direction_dim: str = 'wind_direction',
    renormalize: bool = True,
    weight_sum_tolerance: float = 1e-5,
) -> xr.DataArray:
    bp_da = _bp_dataset_to_direction_da(bp)

    # Prepare distribution: ensure a 'direction' coord with labels matching CARDINAL_8
    if distribution_direction_dim != 'direction':
        distribution = distribution.rename({distribution_direction_dim: 'direction'})
    if 'direction' not in distribution.dims:
        raise ValueError('Distribution must have a direction dimension after renaming.')

    # Attach labels if they are not present (numeric 0..7 assumed in correct order)
    if 'direction' not in distribution.coords or len(distribution['direction']) != 8:
        distribution = distribution.assign_coords(direction=CARDINAL_AND_ORDINAL)

    # Sanity checks
    if set(distribution['direction'].values) != set(CARDINAL_AND_ORDINAL):
        raise ValueError(
            f'Distribution direction labels must match {CARDINAL_AND_ORDINAL}; got {list(distribution["direction"].values)}'
        )

    # distribution = distribution.interp_like(bp_da, method='nearest')
    # --- Interpolate (spatial dims only) BEFORE validity / renorm ---
    spatial_dims = [d for d in bp_da.dims if d != 'direction']
    # Build a dict of target coords that actually appear in distribution
    interp_targets = {d: bp_da[d] for d in spatial_dims if d in distribution.dims}
    if interp_targets:
        distribution = distribution.interp(interp_targets, method='nearest')

    # Identify invalid / missing rows
    any_nan_row = distribution.isnull().any(dim='direction')
    negative_weights = (distribution < 0).any()
    if bool(negative_weights):
        raise ValueError('Distribution contains negative probabilities.')

    row_sum = distribution.sum(dim='direction')

    has_fire_weather = row_sum > 0
    valid_rows = has_fire_weather & (~any_nan_row)

    # Renormalize if requested (only valid rows)
    if renormalize:
        distribution = xr.where(
            valid_rows,
            distribution / row_sum.where(valid_rows),
            distribution,
        )

    post_sum = distribution.sum(dim='direction').where(valid_rows)
    off = (np.abs(post_sum - 1) > weight_sum_tolerance).sum()
    if int(off) > 0:
        warnings.warn(
            f'{int(off)} valid pixel(s) have weight sums outside tolerance {weight_sum_tolerance}.',
            UserWarning,
            stacklevel=2,
        )

    cardinal_and_ordinal_bp = bp_da.sel(direction=CARDINAL_AND_ORDINAL)
    weighted = (cardinal_and_ordinal_bp * distribution).sum(dim='direction')

    weighted = weighted.where(valid_rows)

    weighted.name = 'bp_composite'
    weighted.attrs.update(
        {
            'composition': 'weighted',
            'long_name': 'Weighted composite BP map using wind direction distribution',
            'direction_labels': ALL_9,
            'weights_source': distribution.name or 'wind_direction_distribution',
            'note': ('Rows with no fire-weather hours or NaNs are NaN'),
        }
    )

    return weighted


def _create_weighted_composite_bp_map(
    bp: xr.Dataset,
    wind_direction_distribution: xr.DataArray,
    *,
    include_circular: bool = False,
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
    include_circular : bool, default False
        If True, include the 'circular' kernel layer in the weighted sum. In that
        case it is added with residual weight (1 - sum(probs)) clipped to [0,1].

    Returns
    -------
    xr.DataArray
        Weighted composite burn probability with same spatial dims as inputs.
        Name: 'wind_weighted_bp'. Missing (all-zero) distributions yield NaN.
    """
    direction_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    required = direction_labels + ['circular']
    missing_vars = [v for v in required if v not in bp]
    if missing_vars:
        raise ValueError(
            f'bp dataset missing required directional variables: {missing_vars}. Present: {list(bp.data_vars)}'
        )

    # Build a stacked DataArray of directional layers (exclude circular for now)
    bp_da = (
        bp[direction_labels]
        .to_array(dim='direction')
        .assign_coords(direction=direction_labels)
        .astype('float32')
    )

    # Align / validate distribution ordering
    dist = wind_direction_distribution
    if 'wind_direction' not in dist.dims:
        raise ValueError("wind_direction_distribution must have 'wind_direction' dimension")

    # Try to align coordinate labels if they exist, else assume ordering matches
    if 'wind_direction' in dist.coords and set(dist.wind_direction.values) == set(direction_labels):
        # Reindex to our canonical order
        dist = dist.reindex(wind_direction=direction_labels)
    elif dist.sizes.get('wind_direction') != 8:
        raise ValueError(
            "wind_direction_distribution must have length 8 over 'wind_direction' dimension"
        )

    # Broadcast distribution to spatial dims
    # dist dims: (..., wind_direction). We want to multiply along 'direction'.
    # Rename to match bp_da dim if needed
    if 'direction' != 'wind_direction':
        dist_for_mul = dist.rename({'wind_direction': 'direction'})
    else:
        dist_for_mul = dist

    # Multiply and sum over direction => weighted composite
    weighted = (bp_da * dist_for_mul).sum(dim='direction')

    # Where distribution sums to 0 (no fire-weather hours / no data), set NaN
    dist_sum = dist_for_mul.sum(dim='direction')
    weighted = weighted.where(dist_sum > 0)

    # Optionally incorporate circular kernel with residual probability mass
    if include_circular:
        residual = (1.0 - dist_sum).clip(0.0, 1.0)
        circ = bp['circular'].astype('float32') * residual
        weighted = weighted.fillna(0.0) + circ.where(residual > 0)
        # If residual was the only contributor (dist_sum==0), keep NaN unless residual>0
        weighted = weighted.where(~((dist_sum == 0) & (residual == 0)))

    weighted.name = 'wind_weighted_bp'
    weighted.attrs.update(
        {
            'long_name': 'Burn probability weighted by wind direction distribution',
            'description': 'Per-pixel weighted composite of directional burn probability layers using fire-weather wind direction probabilities.',
            'direction_labels': direction_labels,
            'source_layers': required if include_circular else direction_labels,
            'include_circular': include_circular,
        }
    )
    return weighted


def classify_wind(
    climate_run_subset: xr.Dataset,
    direction_modes_sfc: xr.DataArray,
    rps_30_subset: xr.Dataset,
) -> xr.DataArray:
    from odc.geo.xr import assign_crs

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


def classify_wind_weighted(
    climate_run_subset: xr.Dataset,
    wind_direction_distribution: xr.DataArray,
    rps_30_subset: xr.Dataset,
    *,
    include_circular: bool = False,
) -> xr.DataArray:
    """Classify wind influence using a weighted composite of directional kernels.

    This variant replaces the modal selection with a per-pixel weighted sum of
    directional burn probability layers using the wind direction probability
    distribution (typically conditioned on fire-weather hours / high-index hours).

    Parameters
    ----------
    climate_run_subset : xr.Dataset
        Dataset containing a 'BP' variable (burn probability) to be directionally blurred.
    wind_direction_distribution : xr.DataArray
        Probability distribution over 8 cardinal directions with dimension
        'wind_direction' (length 8) in canonical order ['N','NE','E','SE','S','SW','W','NW'].
    rps_30_subset : xr.Dataset
        Dataset providing target latitude/longitude coordinates used for final alignment.
    include_circular : bool, default False
        If True, includes the 'circular' kernel layer with residual probability mass.

    Returns
    -------
    xr.DataArray
        Weighted composite burn probability aligned to rps_30_subset grid.
    """
    from odc.geo.xr import assign_crs

    blurred_bp = apply_wind_directional_convolution(climate_run_subset['BP'], iterations=3)
    blurred_bp = assign_crs(blurred_bp, crs='EPSG:4326')

    weighted = create_weighted_composite_bp_map(
        blurred_bp, wind_direction_distribution, include_circular=include_circular
    ).drop_vars('direction', errors='ignore')

    weighted = weighted.assign_coords(
        latitude=rps_30_subset.latitude, longitude=rps_30_subset.longitude
    )
    return weighted


def calculate_wind_adjusted_risk(
    *,
    x_slice: slice,
    y_slice: slice,
) -> xr.Dataset:
    # Open input dataset: USFS 30m community risk, USFS 30m interpolated 2011 climate runs and 1/4 degree? ERA5 Wind.
    climate_run_2011 = catalog.get_dataset('2011-climate-run-30m-4326').to_xarray()[['BP']]
    climate_run_2047 = catalog.get_dataset('2047-climate-run-30m-4326').to_xarray()[['BP']]

    rps_30 = catalog.get_dataset('USFS-wildfire-risk-communities-4326').to_xarray()[
        ['BP', 'CRPS', 'RPS']
    ]

    rps_30_subset = rps_30.sel(latitude=y_slice, longitude=x_slice)
    climate_run_2011_subset = climate_run_2011.sel(latitude=y_slice, longitude=x_slice).chunk(
        {'latitude': 6000, 'longitude': 4500}
    )
    climate_run_2047_subset = climate_run_2047.sel(latitude=y_slice, longitude=x_slice).chunk(
        {'latitude': 6000, 'longitude': 4500}
    )

    direction_modes_sfc = (
        catalog.get_dataset('conus404-ffwi-p99-mode-reprojected')
        .to_xarray()
        .wind_direction_mode.sel(latitude=y_slice, longitude=x_slice)
        .load()
    )

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

    fire_risk = (rps_30_subset['RPS']).to_dataset(name='USFS_RPS')

    fire_risk['wind_risk_2011'] = wind_informed_bp_float_corrected_2011 * rps_30_subset['CRPS']
    fire_risk['wind_risk_2047'] = wind_informed_bp_float_corrected_2047 * rps_30_subset['CRPS']

    # Add metadata/attrs to the variables in the dataset
    # BP is burn probability (should be between 0 and 1) and CRPS is the conditional risk to potential structures - aka "if a structure burns, how bad would it be"
    for var in fire_risk.data_vars:
        fire_risk[var].attrs['units'] = 'dimensionless'
        fire_risk[var].attrs['long_name'] = f'{var} (wind-adjusted)'
        fire_risk[var].attrs['description'] = f'{var} adjusted for wind effects'

    return fire_risk.drop_vars(['spatial_ref', 'crs', 'quantile'], errors='ignore')


def nws_fire_weather(
    hurs: xr.DataArray,
    hurs_threshold: float,
    sfcWind: xr.DataArray,
    wind_threshold: float,
    tas: xr.DataArray | None = None,
    tas_threshold: float | None = None,
) -> xr.DataArray:
    """
    calculation of whether or not a day counts as fire weather
    based upon relative humidity, windspeed, temperature and thresholds associated
    with each
    """
    # TODO: use pint-xarray?
    # relative_humidity < 25%
    mask_hurs = hurs < hurs_threshold
    # windspeed > 15 mph
    mps_to_mph_conversion = (
        1000 * 3600 / (25.4 * 12 * 5280)
    )  # convert m/s to mph and then double to account for sustained winds
    mask_sfcWind = (sfcWind * mps_to_mph_conversion) > wind_threshold
    # temperature > 75 deg F
    mask_tas = None  # Initialize with default value
    if tas is not None and tas_threshold is not None:
        mask_tas = ((tas - 273.15) * 9 / 5 + 32) > tas_threshold  # convert K to deg F
    fire_weather_mask = mask_hurs & mask_sfcWind
    if tas is not None and mask_tas is not None:
        fire_weather_mask = fire_weather_mask & mask_tas
    return fire_weather_mask


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


def fosberg_fire_weather_index(hurs: xr.DataArray, T2: xr.DataArray, sfcWind: xr.DataArray):
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
    xr.Dataset
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

    return wind_direction_hist.to_dataset().chunk({'x': -1, 'y': -1})


def compute_modal_wind_direction(distribution: xr.DataArray):
    """
    Compute the modal wind direction from the wind direction distribution.

    Parameters
    ----------
    distribution : xr.DataArray
        Wind direction distribution.

    Returns
    -------
    xr.Dataset
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
