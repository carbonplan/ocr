from unittest.mock import MagicMock, patch

import pydantic
import pytest

from ocr.datasets import Catalog, Dataset, catalog, datasets

# ============= Dataset Tests =============


@pytest.fixture
def sample_dataset():
    """Fixture to create a sample dataset for testing."""
    return Dataset(
        name='test-dataset',
        description='Test Dataset',
        bucket='test-bucket',
        prefix='test-prefix',
        data_format='zarr',
    )


def test_dataset_initialization(sample_dataset):
    """Test Dataset initialization with valid parameters."""
    assert sample_dataset.name == 'test-dataset'
    assert sample_dataset.description == 'Test Dataset'
    assert sample_dataset.bucket == 'test-bucket'
    assert sample_dataset.prefix == 'test-prefix'
    assert sample_dataset.data_format == 'zarr'


def test_dataset_validation():
    """Test that Dataset properly validates data_format."""
    with pytest.raises(pydantic.ValidationError):
        Dataset(
            name='test-dataset',
            description='Test Description',
            bucket='test-bucket',
            prefix='test-prefix',
            data_format='invalid_format',
        )


def test_to_xarray_non_zarr():
    """Test to_xarray raises error when data_format is not zarr."""
    dataset = Dataset(
        name='test-dataset',
        description='Test Dataset',
        bucket='test-bucket',
        prefix='test-prefix',
        data_format='geoparquet',
    )
    with pytest.raises(ValueError, match="Dataset must be in 'zarr' format"):
        dataset.to_xarray(xarray_open_kwargs={})


@patch('duckdb.sql')
def test_query_geoparquet(mock_duckdb_sql, sample_dataset):
    """Test the query_geoparquet method."""
    geoparquet_dataset = Dataset(
        name=sample_dataset.name,
        description=sample_dataset.description,
        bucket=sample_dataset.bucket,
        prefix=sample_dataset.prefix,
        data_format='geoparquet',
    )

    # Mock the DuckDB SQL execution
    mock_result = MagicMock()
    mock_duckdb_sql.return_value = mock_result

    # Test 1: Default query (no custom query provided)
    result = geoparquet_dataset.query_geoparquet()
    mock_duckdb_sql.assert_called_with(
        f"SELECT * FROM read_parquet('s3://{geoparquet_dataset.bucket}/{geoparquet_dataset.prefix}')"
    )
    assert result == mock_result

    # Reset mock
    mock_duckdb_sql.reset_mock()

    # Test 2: Custom query without s3_path placeholder
    custom_query = 'SELECT id, name FROM my_table'
    result = geoparquet_dataset.query_geoparquet(query=custom_query)
    mock_duckdb_sql.assert_any_call('INSTALL SPATIAL; LOAD SPATIAL; INSTALL httpfs; LOAD httpfs')
    mock_duckdb_sql.assert_called_with(custom_query)
    assert result == mock_result

    # Reset mock
    mock_duckdb_sql.reset_mock()

    # Test 3: Custom query with s3_path placeholder
    custom_query_with_placeholder = "SELECT * FROM read_parquet('{s3_path}') WHERE id > 100"
    expected_query = f"SELECT * FROM read_parquet('s3://{geoparquet_dataset.bucket}/{geoparquet_dataset.prefix}') WHERE id > 100"
    result = geoparquet_dataset.query_geoparquet(query=custom_query_with_placeholder)
    mock_duckdb_sql.assert_any_call('INSTALL SPATIAL; LOAD SPATIAL; INSTALL httpfs; LOAD httpfs')
    mock_duckdb_sql.assert_called_with(expected_query)
    assert result == mock_result

    # Test 4: Without installing extensions
    mock_duckdb_sql.reset_mock()
    result = geoparquet_dataset.query_geoparquet(install_extensions=False)
    # Should not call the extension installation
    assert 'INSTALL SPATIAL; LOAD SPATIAL; INSTALL httpfs; LOAD httpfs' not in str(
        mock_duckdb_sql.call_args_list
    )
    mock_duckdb_sql.assert_called_once_with(
        f"SELECT * FROM read_parquet('s3://{geoparquet_dataset.bucket}/{geoparquet_dataset.prefix}')"
    )
    assert result == mock_result

    # Test 5: Error when data_format is not geoparquet
    with pytest.raises(ValueError, match="Dataset must be in 'geoparquet' format"):
        sample_dataset.query_geoparquet()  # sample_dataset has zarr format


@pytest.mark.xfail(reason='This test needs to be updated to work')
@pytest.mark.parametrize('geometry_column', ['geometry', 'geom'])
def test_to_geopandas_method(monkeypatch, sample_dataset, geometry_column):
    """Test the to_geopandas method with a geoparquet dataset."""
    geoparquet_dataset = Dataset(
        name=sample_dataset.name,
        description=sample_dataset.description,
        bucket=sample_dataset.bucket,
        prefix=sample_dataset.prefix,
        data_format='geoparquet',
    )

    # Create mocks
    mock_result = MagicMock()
    mock_df = MagicMock()
    mock_result.df.return_value = mock_df
    mock_geo_series = MagicMock()
    mock_df.__getitem__.return_value.apply.return_value = mock_geo_series
    mock_geodf = MagicMock()

    # Patch the CLASS method instead of the instance method
    original_method = Dataset.query_geoparquet

    def mock_query_geoparquet(self, query=None, **kwargs):
        if self == geoparquet_dataset:  # Only affect our test instance
            return mock_result
        return original_method(self, query, **kwargs)

    monkeypatch.setattr(Dataset, 'query_geoparquet', mock_query_geoparquet)
    monkeypatch.setattr('geopandas.GeoDataFrame', lambda *args, **kwargs: mock_geodf)

    # Call the method
    result = geoparquet_dataset.to_geopandas(
        query='SELECT * FROM test', geometry_column=geometry_column
    )

    # Verify the result
    mock_df.__getitem__.assert_called_with(geometry_column)
    assert result == mock_geodf


@patch('icechunk.s3_storage')
@patch('icechunk.Repository.open')
@patch('xarray.open_dataset')
def test_to_xarray_with_icechunk(
    mock_open_dataset, mock_repo_open, mock_s3_storage, sample_dataset
):
    """Test to_xarray method with icechunk=True."""

    mock_storage = MagicMock()
    mock_repo = MagicMock()
    mock_session = MagicMock()
    mock_ds = MagicMock()

    mock_s3_storage.return_value = mock_storage
    mock_repo_open.return_value = mock_repo
    mock_repo.readonly_session.return_value = mock_session
    mock_open_dataset.return_value = mock_ds

    result = sample_dataset.to_xarray(is_icechunk=True, xarray_open_kwargs={})

    mock_s3_storage.assert_called_once_with(bucket='test-bucket', prefix='test-prefix')
    mock_repo_open.assert_called_once_with(storage=mock_storage)
    mock_repo.readonly_session.assert_called_once_with('main')
    mock_open_dataset.assert_called_once()
    assert result == mock_ds


@patch('xarray.open_dataset')
def test_to_xarray_without_icechunk(mock_open_dataset, sample_dataset):
    """Test to_xarray method with icechunk=False."""
    mock_ds = MagicMock()
    mock_open_dataset.return_value = mock_ds

    storage_options = {'anon': True}
    open_kwargs = {'engine': 'zarr', 'chunks': 'auto'}

    result = sample_dataset.to_xarray(
        is_icechunk=False, xarray_open_kwargs=open_kwargs, xarray_storage_options=storage_options
    )

    mock_open_dataset.assert_called_once_with(
        f's3://{sample_dataset.bucket}/{sample_dataset.prefix}',
        **open_kwargs,
        storage_options=storage_options,
    )
    assert result == mock_ds


# ============= Catalog Tests =============


@pytest.fixture
def sample_catalog(sample_dataset):
    """Fixture to create a sample catalog for testing."""
    return Catalog(datasets=[sample_dataset])


def test_catalog_initialization(sample_catalog, sample_dataset):
    """Test Catalog initialization."""
    assert len(sample_catalog.datasets) == 1
    assert sample_catalog.datasets[0] == sample_dataset


def test_get_dataset_method(sample_dataset):
    """Test get_dataset method of Catalog."""
    dataset1 = sample_dataset
    dataset2 = Dataset(
        name='dataset2',
        description='Dataset 2',
        bucket='bucket2',
        prefix='prefix2',
        data_format='zarr',
    )

    catalog = Catalog(datasets=[dataset1, dataset2])

    # Test finding existing datasets
    assert catalog.get_dataset('test-dataset', version='v1') == dataset1
    assert catalog.get_dataset('dataset2', version='v1') == dataset2

    # Test with non-existent dataset
    with pytest.raises(KeyError):
        catalog.get_dataset('non-existent', version='v1')


def test_catalog_iteration(sample_catalog, sample_dataset):
    """Test iteration over Catalog."""
    datasets_from_iter = list(iter(sample_catalog))
    assert len(datasets_from_iter) == 1
    assert datasets_from_iter[0] == sample_dataset


def test_catalog_str_calls_repr(sample_catalog):
    """Test that str() calls repr()."""
    with patch.object(Catalog, '__repr__', return_value='mocked repr'):
        assert str(sample_catalog) == 'mocked repr'


@patch('rich.console.Console')
@patch('rich.table.Table')
def test_catalog_repr_with_rich(mock_table_cls, mock_console_cls, sample_catalog):
    """Test __repr__ with rich library available."""
    mock_table = MagicMock()
    mock_console = MagicMock()
    mock_output = MagicMock()

    mock_table_cls.return_value = mock_table
    mock_console_cls.return_value = mock_console
    mock_console.file = mock_output

    # Create mock modules that will be imported inside __repr__
    mock_rich_console = type('mock_module', (), {'Console': mock_console_cls})
    mock_rich_table = type('mock_module', (), {'Table': mock_table_cls})

    # Patch the actual modules that get imported in the __repr__ method
    with patch.dict(
        'sys.modules',
        {'rich': MagicMock(), 'rich.console': mock_rich_console, 'rich.table': mock_rich_table},
    ):
        _ = repr(sample_catalog)

    # Verify the table was created with correct title
    mock_table_cls.assert_called_once()
    assert 'OCR Dataset Catalog' in mock_table_cls.call_args[1]['title']

    # Verify columns were added
    assert mock_table.add_column.call_count > 0


def test_catalog_repr_without_rich(sample_catalog):
    """Test __repr__ fallback when rich is not available."""
    with patch('builtins.__import__', side_effect=ImportError()):
        with patch.dict('sys.modules', {'rich': None}):
            repr_result = repr(sample_catalog)

    assert '📊 OCR Dataset Catalog' in repr_result
    assert 'test-dataset' in repr_result


# ============= Module-level Tests =============


def test_module_datasets():
    """Test the module-level datasets list."""
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert all(isinstance(ds, Dataset) for ds in datasets)
    assert all(ds.name for ds in datasets)
    assert all(ds.description for ds in datasets)
    assert all(ds.bucket for ds in datasets)
    assert all(ds.prefix for ds in datasets)
    assert all(ds.data_format for ds in datasets)


def test_module_catalog():
    """Test the module-level catalog instance."""
    assert len(catalog.datasets) >= 1
    assert isinstance(catalog.get_dataset('2011-climate-run', version='v1'), Dataset)
