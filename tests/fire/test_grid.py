import numpy as np
import pytest

from ocr.risks.fire.grid import ChunkingConfig


@pytest.fixture
def sample_chunking_config():
    """Fixture for ChunkingConfig with mocked dependencies."""
    return ChunkingConfig()


# ============= ChunkingConfig Tests =============


@pytest.mark.integration  # reads Source Coop (network)
class TestChunkingConfig:
    """Test cases for ChunkingConfig class."""

    def test_default_initialization(self):
        """Test ChunkingConfig initialization with default values."""
        config = ChunkingConfig()

        # Test that chunks are set based on actual dataset
        assert isinstance(config.chunks, dict)
        assert 'longitude' in config.chunks
        assert 'latitude' in config.chunks
        assert isinstance(config.chunks['longitude'], int)
        assert isinstance(config.chunks['latitude'], int)

    def test_custom_chunks_initialization(self):
        """Test ChunkingConfig with custom chunk values."""
        custom_chunks = {'longitude': 500, 'latitude': 1000}
        config = ChunkingConfig(chunks=custom_chunks)
        assert config.chunks == custom_chunks

    def test_ds_property(self):
        """Test the ds cached property."""
        config = ChunkingConfig()
        result = config.ds

        # Test that it returns an xarray dataset
        import xarray as xr

        assert isinstance(result, xr.Dataset)

        # Test that it has the expected structure
        assert 'CRPS' in result.data_vars
        assert 'longitude' in result.coords
        assert 'latitude' in result.coords

        # Check that EPSG code is correct
        # We're using odc-geo instead of rioxarray here.
        assert result.odc.crs.to_epsg() == 4326

    def test_extent_property(self):
        """Test the extent cached property."""
        config = ChunkingConfig()
        extent = config.extent

        # Test that extent is a shapely box
        from shapely.geometry import box

        assert isinstance(extent, type(box(0, 0, 1, 1)))

        # Test that bounds are reasonable for CONUS
        bounds = extent.bounds
        assert bounds[0] < bounds[2]  # minx < maxx
        assert bounds[1] < bounds[3]  # miny < maxy
        assert -180 <= bounds[0] <= 180  # longitude within valid range
        assert -90 <= bounds[1] <= 90  # latitude within valid range

    def test_transform_property(self):
        """Test the transform cached property."""
        config = ChunkingConfig()
        transform = config.transform

        # Test that transform is not None and has expected properties
        assert transform is not None
        # Transform should have some basic affine transformation properties
        assert hasattr(transform, '__getitem__') or hasattr(transform, '__mul__')

    def test_chunk_info_property(self):
        """Test the chunk_info cached property."""
        config = ChunkingConfig()
        chunk_info = config.chunk_info

        assert isinstance(chunk_info, dict)
        assert 'y_chunks' in chunk_info
        assert 'x_chunks' in chunk_info
        assert 'y_starts' in chunk_info
        assert 'x_starts' in chunk_info

        assert isinstance(chunk_info['y_starts'], np.ndarray)
        assert isinstance(chunk_info['x_starts'], np.ndarray)

        assert len(chunk_info['y_starts']) > 0
        assert len(chunk_info['x_starts']) > 0

    def test_valid_region_ids_property(self):
        """Test the valid_region_ids cached property."""
        config = ChunkingConfig()
        region_ids = config.valid_region_ids

        assert isinstance(region_ids, list)
        assert len(region_ids) > 0
        assert all(isinstance(rid, str) for rid in region_ids)
        assert 'y1_x21' in region_ids
        assert 'y2_x20' in region_ids

    def test_index_to_coords(self):
        """Test the index_to_coords method."""
        config = ChunkingConfig()

        # Test with some realistic indices
        x, y = config.index_to_coords(100, 200)

        # Test that we get numeric coordinates
        assert isinstance(x, int | float)
        assert isinstance(y, int | float)

        # Test that coordinates are in reasonable geographic ranges
        assert -180 <= x <= 180  # longitude range
        assert -90 <= y <= 90  # latitude range

    def test_chunk_id_to_slice(self):
        """Test the chunk_id_to_slice method."""
        config = ChunkingConfig()

        # Use valid indices based on the real dataset structure
        # Real dataset has 17 y chunks and 47 x chunks (indices 0-16 and 0-46)
        y_slice, x_slice = config.chunk_id_to_slice((1, 2))

        # Test that we get slice objects
        assert isinstance(y_slice, slice)
        assert isinstance(x_slice, slice)

        # Test that slices have valid start/stop values
        assert y_slice.start >= 0
        assert y_slice.stop > y_slice.start
        assert x_slice.start >= 0
        assert x_slice.stop > x_slice.start

    def test_chunk_id_to_slice_invalid_bounds(self):
        """Test chunk_id_to_slice with invalid bounds."""
        config = ChunkingConfig()

        # Test negative indices
        with pytest.raises(ValueError, match='Invalid chunk ID'):
            config.chunk_id_to_slice((-1, 0))

        # Test out of bounds indices
        # Real dataset has 17 y chunks (0-16) and 47 x chunks (0-46)
        with pytest.raises(ValueError, match='Invalid chunk ID'):
            config.chunk_id_to_slice((17, 0))  # y index 17 is out of bounds

        with pytest.raises(ValueError, match='Invalid chunk ID'):
            config.chunk_id_to_slice((0, 47))  # x index 47 is out of bounds

    def test_get_chunk_mapping(self):
        """Test the get_chunk_mapping method."""
        config = ChunkingConfig()
        mapping = config.get_chunk_mapping()

        # Test structure
        assert isinstance(mapping, dict)

        # Test that y0_x0 exists (should always exist)
        assert 'y0_x0' in mapping
        assert mapping['y0_x0'] == (0, 0)

        # Test that we have the expected number of mappings
        # Should be 17 * 47 = 799 mappings
        assert len(mapping) == 799

        # Test some valid region IDs from the actual valid_region_ids
        assert 'y1_x3' in mapping
        assert 'y2_x2' in mapping

    def test_region_id_chunk_lookup(self):
        """Test the region_id_chunk_lookup method."""
        config = ChunkingConfig()

        # Test with a known valid region ID
        result = config.region_id_chunk_lookup('y0_x0')

        # Should return a tuple of (y_chunk, x_chunk)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result == (0, 0)

        # Test with another valid region ID if it exists
        if 'y1_x3' in config.valid_region_ids:
            result = config.region_id_chunk_lookup('y1_x3')
            assert isinstance(result, tuple)
            assert result == (1, 3)

    def test_region_id_to_latlon_slices(self):
        """Test the region_id_to_latlon_slices method."""
        config = ChunkingConfig()

        # Test with a known valid region ID
        lat_slice, lon_slice = config.region_id_to_latlon_slices('y0_x0')

        # Should return slice objects
        assert isinstance(lat_slice, slice)
        assert isinstance(lon_slice, slice)

        # Should have numeric start and stop values
        assert isinstance(lat_slice.start, int | float)
        assert isinstance(lat_slice.stop, int | float)
        assert isinstance(lon_slice.start, int | float)
        assert isinstance(lon_slice.stop, int | float)

        # Values should be within reasonable geographic bounds
        assert -90 <= lat_slice.start <= 90
        assert -90 <= lat_slice.stop <= 90
        assert -180 <= lon_slice.start <= 180
        assert -180 <= lon_slice.stop <= 180

    def test_repr(self):
        """Test the __repr__ method."""
        config = ChunkingConfig()
        result = repr(config)

        # Should return a string representation
        assert isinstance(result, str)
        # Should contain some representation of the extent
        assert len(result) > 0
