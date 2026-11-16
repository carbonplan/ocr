from functools import lru_cache

import pyproj
import xarray as xr


@lru_cache
def _get_transformers(wkt: str):
    crs_proj = pyproj.CRS.from_wkt(wkt)
    crs_geo = pyproj.CRS.from_epsg(4326)
    fwd = pyproj.Transformer.from_crs(crs_geo, crs_proj, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs_proj, crs_geo, always_xy=True)
    return fwd, inv


def geo_sel(
    ds: xr.Dataset,
    *,
    lon: float | None = None,
    lat: float | None = None,
    bbox: tuple[float, float, float, float] | None = None,  # (west, south, east, north)
    method: str = 'nearest',
    tolerance: float | None = None,
    crs_wkt: str | None = None,
):
    """
    Geographic selection helper.

    Exactly one of:
      - (lon AND lat)
      - bbox=(west, south, east, north)

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with x, y coordinates and a valid 'crs' variable with WKT
    lon : float, optional
        Longitude of point to select, by default None
    lat : float, optional
        Latitude of point to select, by default None
    bbox : tuple, optional
        Bounding box to select (west, south, east, north), by default None
    method : str, optional
        Method to use for point selection, by default 'nearest'
    tolerance : float, optional
        Tolerance (in units of the dataset's CRS) for point selection, by default None
    crs_wkt : str, optional
        WKT string for the dataset's CRS. If None, attempts to read from ds.crs.attrs['crs_wkt'].

    Returns
    -------
    xarray.Dataset
        Single point: time dimension only
        BBox: retains y, x subset
    """
    if crs_wkt is None:
        try:
            wkt = ds.crs.attrs['crs_wkt']
        except KeyError:
            raise ValueError(
                'CRS WKT not found in dataset attributes. Please provide crs_wkt argument.'
            )
    else:
        wkt = crs_wkt
    fwd, _ = _get_transformers(wkt)

    # --- Case 1: Bounding box ---
    if bbox is not None:
        if any(v is not None for v in (lon, lat)):
            raise ValueError('Provide either bbox OR a point, not both.')
        west, south, east, north = bbox
        # Project the 4 corners
        xs, ys = zip(
            *[
                fwd.transform(west, south),
                fwd.transform(east, south),
                fwd.transform(east, north),
                fwd.transform(west, north),
            ]
        )
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Handle coordinate order (assumes monotonic x and y)
        x_asc = ds.x[0] < ds.x[-1]
        y_asc = ds.y[0] < ds.y[-1]

        x_slice = slice(x_min, x_max) if x_asc else slice(x_max, x_min)
        y_slice = slice(y_min, y_max) if y_asc else slice(y_max, y_min)

        return ds.sel(x=x_slice, y=y_slice)

    # --- Case 2: Single point ---
    if lon is not None and lat is not None:
        x_pt, y_pt = fwd.transform(lon, lat)
        out = ds.sel(x=x_pt, y=y_pt, method=method, tolerance=tolerance)
        # Optional: add requested lon/lat as attributes
        out.attrs['requested_lon'] = float(lon)
        out.attrs['requested_lat'] = float(lat)
        return out

    raise ValueError('You must supply either (lon & lat) or bbox.')
