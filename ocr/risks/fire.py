import typing
import warnings
from math import asin, cos, radians, sin, sqrt

import numpy as np
import xarray as xr
from odc.geo.xr import assign_crs, xr_reproject

from ocr import catalog
from ocr.utils import geo_sel

CARDINAL_AND_ORDINAL = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    Used from https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371 * 10**3  # Radius of earth in meters
    return c * r


def calc_latlon_values(da):
    # grab the central latitude for the region in question. this will inform the scaling of
    # the size of the filter
    center_latitude_index = len(da.latitude.values) // 2
    center_longitude_index = len(da.longitude.values) // 2

    latitude = da.latitude.values[center_latitude_index]
    longitude = da.latitude.values[center_longitude_index]
    latitude_increment = da.latitude.values[1] - da.latitude.values[0]
    longitude_increment = da.longitude.values[1] - da.longitude.values[0]
    return latitude, longitude, latitude_increment, longitude_increment


def generate_weights(
    method: typing.Literal['skewed', 'circular_focal_mean'] = 'skewed',
    kernel_size: float = 81.0,
    circle_diameter: float = 35.0,
    direction: str = 'W',
    latitude_distance: float = 34,
    longitude_distance: float = 25,
) -> np.ndarray:
    """Generate a 2D array of weights for a kernel.

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
        print('herenew5!')
        # elliptical kernel oriented with the major axis horizontal and shifted
        # to the left. this captures pixels to the left, which, in our approach
        # corresponds to a wind from the west
        # this filter is operating on arrays in geographic space (aka each pixel
        # is a length of a set unit of latitude and longitude). this means that
        # as latitude increases, longitude
        # the combined distance of the semi major axis and the shift should scale
        # with latitude
        a = 400  # semi-major axis in meters
        b = 200  # semi-minor axis in meters

        # distance (meters) of one unit of longitude coordinate at this latitude in this dataset
        # distance (meters) of one unit of latitude in this dataset
        # lay out the coordinates of the filter using the approximate distance for this dataset
        # at this latitude.
        x = (
            np.linspace(-kernel_size // 2, kernel_size // 2 + 1, int(kernel_size))
            * longitude_distance
        )
        y = (
            np.linspace(-kernel_size // 2, kernel_size // 2 + 1, int(kernel_size))
            * latitude_distance
        )
        xx, yy = np.meshgrid(x, y)

        # each cardinal/ordinal direction will have an assigned center and rotation_scaler
        # ellipse_specs = {'W': {'center_x': shift, 'center_y': 0, 'rotation_scaler': 2}}
        # # new
        # direction='W'
        # ellipse = ((((xx - ellipse_specs[direction]['center_x']) ** 2) / (2 * a ** 2)) + (((yy-ellipse_specs[direction]['center_y']) ** 2 / (2 * b ** 2)))) <= 1
        # weights = ellipse
        # for cardinal directions we use a simple ellipse
        shift = 110
        diagonal_shift = shift / np.sqrt(2)
        centers = {
            'W': (-shift, 0),
            'E': (shift, 0),
            'N': (0, shift),
            'S': (0, -shift),
            'NE': (diagonal_shift, diagonal_shift),
            'NW': (-diagonal_shift, diagonal_shift),
            'SE': (diagonal_shift, -diagonal_shift),
            'SW': (-diagonal_shift, -diagonal_shift),
        }
        axes = {
            'W': (a, b),
            'E': (a, b),
            'N': (b, a),
            'S': (b, a),
            'NE': (a, b),
            'NW': (b, a),
            'SE': (b, a),
            'SW': (a, b),
        }
        xcenter = centers[direction][0]
        ycenter = centers[direction][1]
        major_axis = axes[direction][0]
        minor_axis = axes[direction][1]
        if direction in ['N', 'S', 'E', 'W']:
            # shift such that the total distance from pixel in question is 510 m
            ellipse = (
                (((xx - xcenter) ** 2) / (major_axis**2))
                + (((yy - ycenter) ** 2) / (minor_axis**2))
            ) <= 1
        # if an ordinal direction then we use a rotated
        # equation for ellipse rotated and shifted:
        elif direction in ['NE', 'NW', 'SW', 'SE']:
            ellipse = (((xx - xcenter) + (yy - ycenter)) ** 2) / (2 * (major_axis**2)) + (
                ((yy - ycenter) - (xx - xcenter)) ** 2
            ) / (2 * (minor_axis**2)) <= 1

        weights = ellipse

        # this worked when shift was positive
        # ellipse = ((((xx + shift) ** 2) / (rotation_scaler * (a ** 2))) + (yy / (rotation_scaler * ( b ** 2)))) <= 1

        # Normalize to sum to 1.0 if there are any non-zero entries
        # weights = weights.astype(np.float32)
        # s = float(weights.sum())
        # if s > 0:
        #     weights = weights / s

    else:
        raise ValueError(f'Unknown method: {method}')

    return weights


def generate_wind_directional_kernels(
    kernel_size: float = 81.0,
    circle_diameter: float = 35.0,
    latitude: float = 38.0,
    longitude: float = -100,
    longitude_increment: float = 0.0003,
    latitude_increment: float = 0.0003,
) -> dict[str, np.ndarray]:
    """Generate a dictionary of 2D arrays of weights for elliptical kernels oriented in different directions.

    Parameters
    ----------
    kernel_size : float, optional
        The size of the kernel, by default 81.0
    circle_diameter : float, optional
        The diameter of the circle, by default 35.0

    Returns
    -------
    kernels : dict[str, np.ndarray]
        A dictionary of 2D arrays of weights for elliptical kernels oriented in different directions.
    """
    weights_dict = {}
    np.arange(0, 360, 45)
    wind_direction_labels = ['W', 'NW', 'N', 'NE', 'E', 'SE', 'S', 'SW']
    # distance (meters) along latitude across longitudes
    latitude_distance = haversine(longitude, latitude, longitude + longitude_increment, latitude)
    # distance (meters) along longitude across longitudes
    longitude_distance = haversine(longitude, latitude, longitude, latitude + latitude_increment)
    # latitude distance in meters
    print(latitude_distance)  #
    print(longitude_distance)

    for direction in wind_direction_labels:
        # our base kernel is oriented to the west
        base = generate_weights(
            method='skewed',
            kernel_size=kernel_size,
            circle_diameter=circle_diameter,
            latitude_distance=latitude_distance,
            longitude_distance=longitude_distance,
            direction=direction,
        ).astype(np.float32)

        weights_dict[direction] = base

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
    latitude: float = 34.0,
    longitude: float = 100.0,
    latitude_increment: float = 0.0003,
    longitude_increment: float = 0.0003,
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

    weights_dict = generate_wind_directional_kernels(
        kernel_size=kernel_size,
        circle_diameter=circle_diameter,
        latitude=latitude,
        longitude=longitude,
        latitude_increment=latitude_increment,
        longitude_increment=longitude_increment,
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
        # nan_mask will keep track of where the nans were to begin with. we'll use it at
        # the very end of the processing
        nan_mask = np.isnan(arr)
        # the cv.filter2D routine doesn't have an option to ignore nans, so it will propagate them.
        # due to reprojections within this processing we are almost guaranteed to have nans in the corners of regions,
        # and those can get propagated very far amidst multiple iterations of convolutions!
        # so, we will cast them to zeros for this step.
        # two notes:
        # 1) this will NOT influence our results, which only accounts for positive values,
        # because any NaNs will be cast into zeros, and so will just be kept track of in the
        # convolved_mask object explained below.
        # 2) we will retain the nans and add them back in using `nan_mask` defined above. this step is inspired by the need
        # to accomodate nans introduced around the corners of each region we are working in as a result of
        # reprojection/cropping issues. we have steps in the `calculated_wind_adjusted_risk` function
        # which crop out the expected bad data
        # (e.g. `riley_2011_30m_4326_subset = riley_2011_30m_4326_subset.sel(latitude=y_slice, longitude=x_slice)`
        # however, in case there are nans introduced for some other reason we want to make sure we don't
        # mistakenly hide those. so, we'll add them back in after all iterations of the convolution are done
        # replace NaNs with 0 in the array before convolution
        arr = np.nan_to_num(arr, nan=0.0)
        for _ in range(iterations):
            # valid_mask is wherever we have positive numbers. we only want to spread numbers informed by non-zero numbers - we are
            # performing an additive method here by spreading burn probability. so, we keep track of which pixels are positive, or valid
            # for spreading. valid_mask will get progressively bigger throughout the runs because 0 values will disappear.
            # we initialize valid_mask fresh each iteration.
            valid_mask = (arr > 0).astype(np.float32)
            # convolved_mask is the fraction the contributing area for each pixel which is from non-zero values (if the values is 1 it is entirely
            # valid values and doesn't need to be adjusted, if the value is 0 it is entirely zero values and there are no valid values in the mask
            # if there were only one valid pixel contributing to the kernel it would make a very tiny positive number
            # dividing by the convolved_mask value will essentially extract out the zeros which have been averaged into the
            # convolved_arr.
            convolved_mask = cv.filter2D(valid_mask, ddepth=-1, kernel=weights)
            # because things can get unstable at very small numbers, we remove the tiny values and cast them to zero.
            # this introduces an assumption that we are not spreading isolated pixels with very low values, which is conservative -
            # this is a slight rounding-down of risk spreading, but the impact is negligible given the low threshold
            # we use for clipping (10e-12).
            # we use a threshold of 10e-12 here. with testing it showed isolated differences (e.g. a handful of spots of a few hundred pixels across
            # a processing region; of maximum magnitude ~-7e-5 in BP) between 10e-12 and 10e-10, so we opt for the lower
            # threshold to clip as little as possible.
            # further, due to numerical precision issues, the method does introduce
            # small negative values which we want to clip, particularly since when we divide by the mask a few lines below,
            # those _tiny_ negative/positive values could propagate

            convolved_mask = np.where(convolved_mask < 10e-12, 0.0, convolved_mask)
            # convolved_arr is the array after having the filter applied to it. any zeros within the filter have been pulled into the averaging.
            convolved_arr = cv.filter2D(arr, ddepth=-1, kernel=weights)
            convolved_arr = np.where(convolved_arr < 10e-12, 0.0, convolved_arr)
            # output_arr is the final result. wherever the convolved_mask is 0 there was no contribution of valid pixels
            # to the convolution, so it should just be zero. wherever convolved_mask is greater than 0 that means there
            # was at least one valid pixel in the filter contributing so it should get a non-zero number! however, we want
            # to extract out all of the zeros that might have supressed that number. we do that by renormalizing by the
            # value of the convolved_mask. for example, if the convolved_mask value for a pixel is 1, it means that all
            # the pixels in that area were non-zero and so the value doesn't need to be adjusted. if the value is 0.5,
            # that means that half of the pixels in that kernel were zeros, so the final convolved pixel value was
            # diluted in half by zeros. to compensate we divide the `convolved_arr` value by the `convolved_mask` value,
            # in this example, dividing by 0.5 and thus dubling the `convoled_arr` value. this is an around-the-way
            # approach to essentially treat those zeros as nans and applying a `ignore_nans` flag!
            # NOTE: we renormalize by the convolved_mask every iteration because we want to make this correction every time
            # as opposed to going back after the fact.
            arr = np.where(convolved_mask > 0, convolved_arr / convolved_mask, 0.0)
        # confirm array is entirely positive. this should be handled by the above clipping but we just want to make sure!
        np.testing.assert_equal((arr < 0).sum(), 0)
        # add back in nans at the very end after iterations done. use the `nan_mask` to do this. this will
        # make sure that any nans are retained, whether they're ones we expect or ones we didn't! and so
        # we'll make sure they can trigger tests later on in the processing steps
        arr = np.where(nan_mask, np.nan, arr)
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

    with xr.set_options(arithmetic_join='exact'):
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
        latitude, longitude, latitude_increment, longitude_increment = calc_latlon_values(
            riley_30m_4326['BP']
        )
        print(
            f'at latitude {latitude} the latitude increment is {latitude_increment} and the longitude_increment is {longitude_increment}'
        )
        blurred_bp_30m_4326 = apply_wind_directional_convolution(
            riley_30m_4326['BP'],
            iterations=3,
            latitude=latitude,
            longitude=longitude,
            latitude_increment=latitude_increment,
            longitude_increment=longitude_increment,
        )
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

    # smooth using a Gaussian filter ~300m radius
    filter_radius = 300 // latitude_increment
    filter_size = (
        2 * filter_radius + 1
    )  # for computational reasons we need our filter to be an array with a odd (not even) dimensions
    print(f'gaussian filter is of size :{filter_size}')
    smoothed_bp = xr.apply_ufunc(
        cv.GaussianBlur,
        wind_informed_bp_combined.chunk(latitude=-1, longitude=-1),
        input_core_dims=[['latitude', 'longitude']],
        output_core_dims=[['latitude', 'longitude']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32],
        kwargs={'ksize': (25, 25), 'sigmaX': 0},
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
    fire_risk['rps_scott'] = rps_30_subset['RPS']
    fire_risk['rps_scott'].attrs['description'] = (
        'Annual risk to potential structures from Scott et al., (2024)'
    )
    fire_risk['rps_scott'].attrs['units'] = 'percent'
    with xr.set_options(arithmetic_join='exact'):
        # rps_2011 (our wind-informed RPS value)
        fire_risk['rps_2011'] = wind_informed_bp_combined_2011 * rps_30_subset['CRPS']
        fire_risk['rps_2011'].attrs['description'] = (
            'Annual relative risk to potential structures (RPS) for ~2011 climate conditions. Calculated as bp_2011 × crps_scott'
        )
        fire_risk['rps_2011'].attrs['units'] = 'percent'
        # rps_2047 (our wind-informed RPS value)
        fire_risk['rps_2047'] = wind_informed_bp_combined_2047 * rps_30_subset['CRPS']
        fire_risk['rps_2047'].attrs['description'] = (
            'Annual risk to potential structures (RPS) for ~2047 climate conditions. Calculated as bp_2047 × crps_scott'
        )
        fire_risk['rps_2047'].attrs['units'] = 'percent'

        # bp_2011 (our wind-informed BP value)
        fire_risk['bp_2011'] = wind_informed_bp_combined_2011
        fire_risk['bp_2011'].attrs['description'] = (
            'Annual burn probability for ~2011 climate conditions'
        )

        # bp_2047 (our wind-informed BP value)
        fire_risk['bp_2047'] = wind_informed_bp_combined_2047
        fire_risk['bp_2047'].attrs['description'] = (
            'Annual burn probability for ~2047 climate conditions'
        )

        # crps_scott (from USFS Scott 2024)(RDS-2020-0016-2)
        fire_risk['crps_scott'] = rps_30_subset['CRPS']
        fire_risk['crps_scott'].attrs['description'] = (
            'Conditional risk to potential structures (cRPS) from Scott et al., (2024)'
        )
        fire_risk['crps_scott'].attrs['units'] = 'percent'

        # bp_2011_riley (BP from Riley 2025 (RDS-2025-0006))
        fire_risk['bp_2011_riley'] = riley_2011_30m_4326_subset['BP']
        fire_risk['bp_2011_riley'].attrs['description'] = (
            'Burn probability for ~2011 from Riley et al. (2025) (RDS-2025-0006)'
        )

        # bp_2047_riley (BP from Riley 2025 (RDS-2025-0006))
        fire_risk['bp_2047_riley'] = riley_2047_30m_4326_subset['BP']
        fire_risk['bp_2047_riley'].attrs['description'] = (
            'Burn probability for ~2047 from Riley et al. (2025) (RDS-2025-0006)'
        )

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
