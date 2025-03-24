import io
import sys
from unittest import mock

from ocr.dep_versions import show_versions


def test_show_versions_handles_import_error() -> None:
    """Test that show_versions properly handles modules that can't be imported."""
    output = io.StringIO()

    # Patch importlib.import_module to raise ImportError for a specific module
    with mock.patch('importlib.import_module') as mock_import:
        mock_import.side_effect = ImportError('Module not found')
        show_versions(file=output)

    # Get output
    result = output.getvalue()

    # At least one module should show None (since all imports will fail)
    assert ': None' in result


def test_show_versions_with_existing_modules() -> None:
    """Test that show_versions uses already imported modules."""
    output = io.StringIO()

    # Create mock module with version
    mock_module = mock.MagicMock(__version__='2.0.0')

    # Add it to sys.modules
    with mock.patch.dict(sys.modules, {'xarray': mock_module}):
        show_versions(file=output)

    # Check output
    result = output.getvalue()
    assert 'xarray: 2.0.0' in result


def test_show_versions_handles_version_error() -> None:
    """Test that show_versions handles modules that don't have __version__."""
    output = io.StringIO()

    # Create a mock module without __version__
    mock_module = mock.MagicMock(spec=[])  # No __version__ attribute

    # Add it to sys.modules
    with mock.patch.dict(sys.modules, {'scipy': mock_module}):
        show_versions(file=output)

    # Check output
    result = output.getvalue()
    assert 'scipy: installed' in result
