from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from ocr.cal_fire_dins import load_structures_destroyed


class MockDuckDBRelation:
    def __init__(self, data):
        self.data = data

    def df(self):
        return self.data


@pytest.fixture
def mock_catalog():
    mock_dataset = Mock()
    mock_dataset.query_geoparquet.side_effect = (
        lambda query, **kwargs: MockDuckDBRelation(
            pd.DataFrame(
                {'INCIDENTNAME': ['Eaton', 'Dixie', 'Camp', 'Tubbs', 'LNU Lightning Complex']}
            )
        )
        if 'DISTINCT INCIDENTNAME' in query
        else MockDuckDBRelation(
            pd.DataFrame(
                {
                    'INCIDENTNAME': ['Eaton', 'Eaton'],
                    'DAMAGE': ['Destroyed (>50%)', 'Minor'],
                    'COUNTY': ['Butte', 'Butte'],
                    'GEOMETRY': [Point(1, 1), Point(2, 2)],
                }
            )
        )
    )

    mock_dataset.to_geopandas.side_effect = lambda query, **kwargs: gpd.GeoDataFrame(
        {
            'INCIDENTNAME': ['Eaton', 'Eaton'],
            'DAMAGE': ['Destroyed (>50%)', 'Minor'],
            'COUNTY': ['Butte', 'Butte'],
            'geometry': [Point(1, 1), Point(2, 2)],
        },
        geometry='geometry',
    )

    return mock_dataset


@patch('ocr.cal_fire_dins.catalog')
def test_load_structures_destroyed_success(mock_catalog_module, mock_catalog):
    # Set up the mock catalog
    mock_catalog_module.get_dataset.return_value = mock_catalog

    # Call the function
    result = load_structures_destroyed('Eaton')

    # Verify the result
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 2
    assert list(result['INCIDENTNAME']) == ['Eaton', 'Eaton']

    # Verify the correct query was executed
    mock_catalog.to_geopandas.assert_called_once()
    call_args = mock_catalog.to_geopandas.call_args[0][0]
    assert "SELECT * FROM read_parquet('{s3_path}') WHERE INCIDENTNAME = 'Eaton'" in call_args


@patch('ocr.cal_fire_dins.catalog')
def test_load_structures_destroyed_with_columns(mock_catalog_module, mock_catalog):
    # Set up the mock catalog
    mock_catalog_module.get_dataset.return_value = mock_catalog

    # Call the function with specific columns
    columns = ['DAMAGE', 'COUNTY', 'GEOMETRY']
    _ = load_structures_destroyed('Eaton', columns=columns)

    # Verify the correct query was executed with columns
    mock_catalog.to_geopandas.assert_called_once()
    call_args = mock_catalog.to_geopandas.call_args[0][0]
    assert "SELECT DAMAGE, COUNTY, GEOMETRY FROM read_parquet('{s3_path}')" in call_args


@patch('ocr.cal_fire_dins.catalog')
def test_load_structures_destroyed_with_target_crs(mock_catalog_module, mock_catalog):
    # Set up the mock catalog
    mock_catalog_module.get_dataset.return_value = mock_catalog

    # Call the function with target_crs
    _ = load_structures_destroyed('Eaton', target_crs='EPSG:5070')

    # Verify target_crs was passed correctly
    mock_catalog.to_geopandas.assert_called_once()
    assert mock_catalog.to_geopandas.call_args[1]['target_crs'] == 'EPSG:5070'


@patch('ocr.cal_fire_dins.catalog')
def test_load_structures_destroyed_nonexistent_fire(mock_catalog_module, mock_catalog):
    # Set up the mock catalog
    mock_catalog_module.get_dataset.return_value = mock_catalog

    # Test with non-existent fire name
    with pytest.raises(ValueError) as excinfo:
        load_structures_destroyed('NonExistentFire')

    # Verify error message contains the fire name
    assert "Fire 'NonExistentFire' not found" in str(excinfo.value)


@patch('ocr.cal_fire_dins.catalog')
def test_load_structures_destroyed_with_suggestions(mock_catalog_module, mock_catalog):
    # Set up the mock catalog
    mock_catalog_module.get_dataset.return_value = mock_catalog

    # Test with a fire name that's close to an existing one
    with pytest.raises(ValueError) as excinfo:
        load_structures_destroyed('Eatn')

    # Verify error message contains suggestions
    assert 'Did you mean: Eaton?' in str(excinfo.value)


@patch('ocr.cal_fire_dins.get_close_matches', return_value=[])
@patch('ocr.cal_fire_dins.catalog')
def test_load_structures_destroyed_no_suggestions(
    mock_catalog_module, mock_get_close_matches, mock_catalog
):
    # Set up the mock catalog
    mock_catalog_module.get_dataset.return_value = mock_catalog

    # Test with a fire name that has no close matches
    with pytest.raises(ValueError) as excinfo:
        load_structures_destroyed('CompletelyDifferent')

    # Verify error message contains examples
    assert 'Available fires include:' in str(excinfo.value)
