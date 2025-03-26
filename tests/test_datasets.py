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
            data_format='invalid_format',  # Not one of the allowed formats
        )


def test_to_xarray_non_zarr():
    """Test to_xarray raises error when data_format is not zarr."""
    dataset = Dataset(
        name='test-dataset',
        description='Test Dataset',
        bucket='test-bucket',
        prefix='test-prefix',
        data_format='csv',  # Not zarr
    )
    with pytest.raises(ValueError, match="Dataset must be in 'zarr' format"):
        dataset.to_xarray(xarray_open_kwargs={})


@patch('icechunk.s3_storage')
@patch('icechunk.Repository.open')
@patch('xarray.open_dataset')
def test_to_xarray_with_icechunk(
    mock_open_dataset, mock_repo_open, mock_s3_storage, sample_dataset
):
    """Test to_xarray method with icechunk=True."""
    # Setup mocks
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
    open_kwargs = {'chunks': 'auto'}

    result = sample_dataset.to_xarray(
        is_icechunk=False, xarray_open_kwargs=open_kwargs, xarray_storage_options=storage_options
    )

    mock_open_dataset.assert_called_once_with(
        f's3://{sample_dataset.bucket}/{sample_dataset.prefix}',
        **open_kwargs,
        storage_options=storage_options,
    )
    assert result == mock_ds


def test_unimplemented_methods(sample_dataset):
    """Test that unimplemented methods raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        Dataset.to_geopandas()

    with pytest.raises(NotImplementedError):
        Dataset.to_pandas()


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
    assert catalog.get_dataset('test-dataset') == dataset1
    assert catalog.get_dataset('dataset2') == dataset2

    # Test with non-existent dataset
    assert catalog.get_dataset('non-existent') is None


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
    assert mock_table.add_column.call_count == 4


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
    assert len(datasets) == 1
    assert datasets[0].name == '2011-climate-run'
    assert datasets[0].description == 'USFS 2011 Climate Run'
    assert datasets[0].bucket == 'carbonplan-ocr'
    assert datasets[0].prefix == 'input/fire-risk/tensor/USFS/2011ClimateRun_Icechunk'
    assert datasets[0].data_format == 'zarr'


def test_module_catalog():
    """Test the module-level catalog instance."""
    assert len(catalog.datasets) == 1
    assert catalog.datasets[0].name == '2011-climate-run'
    assert catalog.get_dataset('2011-climate-run') is not None
