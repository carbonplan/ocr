from difflib import get_close_matches

import geopandas as gpd
import pydantic

from ocr.datasets import catalog


@pydantic.validate_call
def load_structures_destroyed(
    fire_name: str,
    columns: list[str] = None,
    target_crs: str = None,
) -> gpd.GeoDataFrame:
    """
    Load structures destroyed/damaged data for a specific fire.

    Parameters
    ----------
    fire_name : str
        Name of the fire (e.g., 'Eaton')
    columns : list[str], optional
        List of columns to select. If None, selects all columns.
    target_crs : str, optional
        Target coordinate reference system (e.g., 'EPSG:5070')


    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the structures data for the specified fire

    Raises
    ------
    ValueError
        If the specified fire name doesn't exist in the dataset

    Example
    -------
    >>> # Load data for the Eaton fire and convert to EPSG:5070
    >>> eaton_data = load_structures_destroyed('Eaton', target_crs='EPSG:5070')
    >>>
    >>> # Load only destroyed buildings for the Dixie fire
    >>> dixie_destroyed = load_structures_destroyed('Dixie'')
    """
    dataset = catalog.get_dataset('cal-fire-damage-inspection')

    check_query = "SELECT DISTINCT INCIDENTNAME FROM read_parquet('{s3_path}')"
    result = dataset.query_geoparquet(check_query)
    available_fires = set(result.df()['INCIDENTNAME'].tolist())
    if fire_name not in available_fires:
        suggestions = get_close_matches(fire_name, available_fires, n=3, cutoff=0.6)

        error_msg = f"Fire '{fire_name}' not found in the CAL FIRE dataset."
        if suggestions:
            error_msg += f' Did you mean: {", ".join(suggestions)}?'
        else:
            # Show a few examples of available fires
            sample_fires = sorted(list(available_fires))[:5]
            error_msg += f' Available fires include: {", ".join(sample_fires)}...'

        raise ValueError(error_msg)
    # Build the SQL query
    if columns is None:
        columns_str = '*'
    else:
        columns_str = ', '.join(columns)

    query = (
        f"SELECT {columns_str} FROM read_parquet('{{s3_path}}') WHERE INCIDENTNAME = '{fire_name}'"
    )

    # Execute query and convert to GeoDataFrame
    gdf = dataset.to_geopandas(query, target_crs=target_crs)
    return gdf
