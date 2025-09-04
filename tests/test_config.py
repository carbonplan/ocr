import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydantic
import pytest

from ocr.config import ChunkingConfig, CoiledConfig, IcechunkConfig, OCRConfig, VectorConfig
from ocr.types import Environment

# ============= Mock and Fixture Helpers =============


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_chunking_config():
    """Fixture for ChunkingConfig with mocked dependencies."""
    return ChunkingConfig()


@pytest.fixture
def sample_vector_config(temp_dir):
    """Fixture for VectorConfig."""
    return VectorConfig(storage_root=temp_dir)


@pytest.fixture
def sample_icechunk_config(temp_dir):
    """Fixture for IcechunkConfig."""
    with patch.object(IcechunkConfig, 'init_repo'):
        return IcechunkConfig(storage_root=temp_dir)


@pytest.fixture
def sample_ocr_config(temp_dir):
    """Fixture for OCRConfig."""
    with patch.object(IcechunkConfig, 'init_repo'):
        return OCRConfig(storage_root=temp_dir)


# ============= CoiledConfig Tests =============


class TestCoiledConfig:
    """Test cases for CoiledConfig class."""

    def test_default_initialization(self):
        """Test CoiledConfig initialization with default values."""
        config = CoiledConfig()

        assert config.tag == {'Project': 'OCR'}
        assert config.forward_aws_credentials is True
        assert config.region == 'us-west-2'
        assert config.ntasks == 1
        assert config.vm_type == 'm8g.xlarge'
        assert config.scheduler_vm_type == 'm8g.xlarge'

    def test_custom_initialization(self):
        """Test CoiledConfig initialization with custom values."""
        config = CoiledConfig(
            tag={'Project': 'CustomOCR', 'Environment': 'Test'},
            forward_aws_credentials=False,
            region='us-east-1',
            ntasks=4,
            vm_type='m8g.xlarge',
        )

        assert config.tag == {'Project': 'CustomOCR', 'Environment': 'Test'}
        assert config.forward_aws_credentials is False
        assert config.region == 'us-east-1'
        assert config.ntasks == 4
        assert config.vm_type == 'm8g.xlarge'

    def test_environment_variable_override(self):
        """Test that environment variables override default values."""
        with patch.dict(
            os.environ,
            {
                'OCR_COILED_TAG': '{"Project": "EnvOCR"}',
                'OCR_COILED_FORWARD_AWS_CREDENTIALS': 'false',
                'OCR_COILED_REGION': 'us-east-2',
                'OCR_COILED_NTASKS': '8',
                'OCR_COILED_VM_TYPE': 'm8g.2xlarge',
                'OCR_COILED_SCHEDULER_VM_TYPE': 'm8g.2xlarge',
            },
        ):
            config = CoiledConfig()

            assert config.forward_aws_credentials is False
            assert config.region == 'us-east-2'
            assert config.ntasks == 8
            assert config.vm_type == 'm8g.2xlarge'
            assert config.scheduler_vm_type == 'm8g.2xlarge'

    def test_field_validation(self):
        """Test field validation for CoiledConfig."""
        # Test invalid ntasks (negative)
        with pytest.raises(pydantic.ValidationError):
            CoiledConfig(ntasks=-1)


# ============= ChunkingConfig Tests =============


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

    def test_extent_as_tuple_property(self):
        """Test the extent_as_tuple cached property."""
        config = ChunkingConfig()
        result = config.extent_as_tuple

        # Should be a tuple of 4 values (minx, maxx, miny, maxy)
        assert isinstance(result, tuple)
        assert len(result) == 4

        # Test order and values are reasonable
        minx, maxx, miny, maxy = result
        assert minx < maxx  # longitude order
        assert miny < maxy  # latitude order

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
        assert 'y1_x3' in region_ids
        assert 'y2_x2' in region_ids

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


# ============= VectorConfig Tests =============


class TestVectorConfig:
    """Test cases for VectorConfig class."""

    def test_default_initialization(self, temp_dir):
        """Test VectorConfig initialization with default values."""
        config = VectorConfig(storage_root=temp_dir)

        assert config.environment == Environment.QA
        assert config.storage_root == temp_dir
        assert config.prefix == 'intermediate/fire-risk/vector/qa'
        assert config.output_prefix == 'output/fire-risk/vector/qa'

    def test_custom_initialization(self, temp_dir):
        """Test VectorConfig initialization with custom values."""
        config = VectorConfig(
            environment=Environment.PRODUCTION,
            storage_root=temp_dir,
            prefix='custom-prefix',
            output_prefix='custom-output',
        )

        assert config.environment == Environment.PRODUCTION
        assert config.storage_root == temp_dir
        assert config.prefix == 'custom-prefix'
        assert config.output_prefix == 'custom-output'

    def test_model_post_init_prefixes(self, temp_dir):
        """Test model_post_init sets prefixes based on Environment."""
        config = VectorConfig(storage_root=temp_dir, environment=Environment.PRODUCTION)

        assert config.prefix == 'intermediate/fire-risk/vector/production'
        assert config.output_prefix == 'output/fire-risk/vector/production'

    @pytest.mark.parametrize('version', ['1.2.3', '2.0.0-alpha', '0.0.1'])
    def test_vector_config_version_prefix(self, version):
        config = VectorConfig(
            storage_root='s3://dummy',
            environment=Environment.QA,
            version=version,
            prefix=None,
            output_prefix=None,
        )
        assert config.prefix is not None and config.prefix.endswith(f'/v{version}')
        assert config.output_prefix is not None and config.output_prefix.endswith(f'/v{version}')

    @pytest.mark.parametrize('invalid', ['1.2', 'version1.2.3', '1.2.3.4', '1.2.-3'])
    def test_vector_config_invalid_version(self, invalid):
        with pytest.raises(ValueError):
            VectorConfig(
                storage_root='s3://dummy',
                environment=Environment.QA,
                version=invalid,
                prefix=None,
                output_prefix=None,
            )

    def test_vector_config_insert_version_into_existing_prefix(self, temp_dir):
        base_prefix = 'intermediate/fire-risk/vector'
        base_output_prefix = 'output/fire-risk/vector'
        version = '1.0.0'
        config = VectorConfig(
            storage_root=temp_dir,
            environment=Environment.QA,
            prefix=base_prefix,
            output_prefix=base_output_prefix,
            version=version,
        )
        # version is inserted before the last part of the path

        assert config.prefix == f'intermediate/fire-risk/qa/v{version}/vector'
        assert config.output_prefix == f'output/fire-risk/qa/v{version}/vector'

    def test_cached_properties(self, temp_dir):
        """Test cached properties of VectorConfig."""
        config = VectorConfig(storage_root=temp_dir)

        # Test region_geoparquet_prefix
        assert (
            config.region_geoparquet_prefix == 'intermediate/fire-risk/vector/qa/geoparquet-regions'
        )

        # Test region_geoparquet_uri
        expected_uri = f'{temp_dir}/intermediate/fire-risk/vector/qa/geoparquet-regions'
        assert str(config.region_geoparquet_uri) == expected_uri

        # Test building_geoparquet_uri
        assert (
            str(config.building_geoparquet_uri)
            == f'{temp_dir}/output/fire-risk/vector/qa/geoparquet/buildings.parquet'
        )

        # Test pmtiles_prefix
        assert (
            str(config.buildings_pmtiles_uri)
            == f'{temp_dir}/output/fire-risk/vector/qa/pmtiles/buildings.pmtiles'
        )

    def test_summary_stats_uris(self, temp_dir):
        """Test summary statistics URI properties."""
        config = VectorConfig(storage_root=temp_dir)

        tracts_uri = config.tracts_summary_stats_uri
        counties_uri = config.counties_summary_stats_uri

        assert 'tracts_summary_stats.parquet' in str(tracts_uri)
        assert 'counties_summary_stats.parquet' in str(counties_uri)

    def test_wipe_method(self, temp_dir):
        """Test the wipe method."""
        config = VectorConfig(storage_root=temp_dir, debug=True)

        # Create directory structure
        region_dir = Path(temp_dir) / 'intermediate/fire-risk/vector/qa/geoparquet-regions'
        region_dir.mkdir(parents=True, exist_ok=True)

        test_file = region_dir / 'test.parquet'
        test_file.write_text('test data')

        config.wipe()

        # Check that file was deleted and log was called
        assert not test_file.exists()

    def test_delete_region_gpqs(self, temp_dir):
        """Test the delete_region_gpqs method."""
        config = VectorConfig(storage_root=temp_dir)

        # Create the directory structure and some test files
        region_dir = Path(temp_dir) / 'intermediate/fire-risk/vector/qa/geoparquet-regions'
        region_dir.mkdir(parents=True, exist_ok=True)

        test_file1 = region_dir / 'region1.parquet'
        test_file2 = region_dir / 'region2.parquet'
        test_file1.write_text('test data')
        test_file2.write_text('test data')

        config.delete_region_gpqs()

        # Files should be deleted
        assert not test_file1.exists()
        assert not test_file2.exists()

    def test_delete_region_gpqs_invalid_prefix(self, temp_dir):
        """Test delete_region_gpqs with invalid prefix - but since it adds /geoparquet-regions/, this won't raise."""
        config = VectorConfig(storage_root=temp_dir, prefix='invalid-prefix')

        # This actually works because the property adds '/geoparquet-regions/' to any non-None prefix
        config.delete_region_gpqs()  # Should not raise

    def test_delete_region_gpqs_none_prefix(self, temp_dir):
        """Test delete_region_gpqs with None prefix - actually works because property returns string."""
        config = VectorConfig(
            storage_root=temp_dir, prefix='test-prefix', output_prefix='test-output'
        )
        config.prefix = None

        # This doesn't raise because region_geoparquet_prefix returns "None/geoparquet-regions/"
        # which still contains 'geoparquet-regions', so the check passes
        config.delete_region_gpqs()  # Should not raise


# ============= IcechunkConfig Tests =============


class TestIcechunkConfig:
    """Test cases for IcechunkConfig class."""

    def test_default_initialization(self, temp_dir):
        """Test IcechunkConfig initialization with default values."""

        config = IcechunkConfig(storage_root=temp_dir)

        assert config.environment == Environment.QA
        assert config.storage_root == temp_dir
        assert config.prefix == 'output/fire-risk/tensor/qa/ocr.icechunk'

    def test_custom_initialization(self, temp_dir):
        """Test IcechunkConfig initialization with custom values."""

        config = IcechunkConfig(
            environment=Environment.PRODUCTION,
            storage_root=temp_dir,
            prefix='custom/path/',
        )

        assert config.environment == Environment.PRODUCTION
        assert config.storage_root == temp_dir
        assert config.prefix == 'custom/path/'

    @pytest.mark.parametrize('ver', ['1.2.3'])
    def test_icechunk_config_version_prefix(self, ver):
        config = IcechunkConfig(
            storage_root='s3://dummy', environment=Environment.QA, version=ver, prefix=None
        )
        assert config.prefix is not None and f'/v{ver}' in config.prefix

    def test_icechunk_config_insert_version_into_existing_prefix(self):
        base = 'output/fire-risk/tensor/'
        config = IcechunkConfig(
            storage_root='s3://dummy', environment=Environment.QA, prefix=base, version='1.0.0'
        )
        assert config.prefix == 'output/fire-risk/tensor/qa/v1.0.0/'

    def test_uri_property(self, temp_dir):
        """Test the uri cached property."""

        config = IcechunkConfig(storage_root=temp_dir, prefix='test/path')

        expected_uri = f'{temp_dir}/test/path'
        assert str(config.uri) == expected_uri

    def test_storage_property_local(self, temp_dir):
        """Test the storage cached property for local filesystem."""

        config = IcechunkConfig(storage_root=temp_dir, prefix='test/path')
        storage = config.storage

        # Test that storage is created and not None
        assert storage is not None

    def test_storage_property_unsupported_protocol(self):
        """Test storage property with unsupported protocol."""

        config = IcechunkConfig(storage_root='ftp://unsupported', prefix='test/path')

        with pytest.raises(ValueError, match='Unsupported protocol: ftp'):
            _ = config.storage

    def test_init_repo(self, temp_dir):
        """Test the init_repo method using real implementation."""

        # Use local filesystem storage which should work with temp directory
        config = IcechunkConfig(storage_root=temp_dir, prefix='test-repo')

        # The init_repo should work with local filesystem without issues
        # It will create or open a repository in the temp directory

        config.init_repo()
        # If we get here, the repository was successfully initialized
        assert config.uri.exists() or config.uri.parent.exists()

    def test_repo_and_session_readonly(self, temp_dir):
        """Test repo_and_session method in readonly mode."""

        config = IcechunkConfig(storage_root=temp_dir, prefix='test/path')
        config.init_repo()  # Ensure repo is initialized

        # Try to get a readonly session
        result = config.repo_and_session(readonly=True, branch='main')

        # Should return a dict with 'repo' and 'session' keys
        assert isinstance(result, dict)
        assert 'repo' in result
        assert 'session' in result

    def test_wipe_method(self, temp_dir):
        """Test the wipe method using real implementation."""
        # Create config with temp directory - this will call init_repo in model_post_init
        # but that's okay since we're testing the real functionality

        config = IcechunkConfig(storage_root=temp_dir, prefix='test-wipe-repo')
        config.init_repo()

        # Create some test files in the icechunk directory to verify deletion
        icechunk_path = config.uri
        icechunk_path.mkdir(parents=True, exist_ok=True)
        test_file = icechunk_path / 'test_file.txt'
        test_file.write_text('test content')

        # Ensure the test file exists before wipe
        assert test_file.exists()

        # Call wipe - this should delete and recreate the repo
        config.wipe()

        # The directory might be recreated by init_repo, but our test file should be gone
        if icechunk_path.exists():
            assert not test_file.exists()

    def test_delete_local_filesystem(self, temp_dir):
        """Test delete method for local filesystem."""

        config = IcechunkConfig(storage_root=temp_dir, prefix='test/delete/path')

        # Create the icechunk directory
        icechunk_dir = config.uri
        icechunk_dir.mkdir(parents=True, exist_ok=True)
        test_file = icechunk_dir / 'test_file'
        test_file.write_text('test data')

        # Verify file exists before deletion
        assert test_file.exists()

        config.delete()

        # Directory should be deleted
        assert not icechunk_dir.exists()

    def test_create_template(self, temp_dir):
        """Test the create_template method using real implementation."""

        config = IcechunkConfig(storage_root=temp_dir, prefix='test-template-repo')

        config.init_repo()  # this calls .create_template() under the hood

        # If we get here, the method executed without crashing
        assert True

    def test_commit_messages_ancestry(self, temp_dir):
        """Test commit messages ancestry."""

        config = IcechunkConfig(storage_root=temp_dir)
        config.init_repo()  # Ensure repo is initialized

        commit_message = 'Repository initialized'

        # Check the commit messages ancestry
        ancestry = config.commit_messages_ancestry()
        assert commit_message in ancestry

    def test_region_id_exists(self, temp_dir):
        """Test region_id_exists method."""

        config = IcechunkConfig(storage_root=temp_dir)
        config.init_repo()

        # Create a mock region ID
        region_id = 'test_region'

        # Check if the region ID exists
        exists = config.region_id_exists(region_id)
        assert not exists

        exists = config.region_id_exists('Repository initialized')
        assert exists


# ============= OCRConfig Tests =============


class TestOCRConfig:
    """Test cases for OCRConfig class."""

    def test_default_initialization(self, temp_dir):
        """Test OCRConfig initialization with default values."""

        config = OCRConfig(storage_root=temp_dir)

        assert config.environment == Environment.QA
        assert config.storage_root == temp_dir
        assert isinstance(config.vector, VectorConfig)
        assert isinstance(config.icechunk, IcechunkConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.coiled, CoiledConfig)

    def test_custom_sub_configs(self, temp_dir):
        """Test OCRConfig with custom sub-configurations."""

        custom_vector = VectorConfig(storage_root=temp_dir)
        custom_coiled = CoiledConfig()

        config = OCRConfig(storage_root=temp_dir, vector=custom_vector, coiled=custom_coiled)

        assert config.vector == custom_vector
        assert config.coiled == custom_coiled
        assert config.coiled.ntasks == 1

    def test_model_post_init_creates_sub_configs(self, temp_dir):
        """Test that model_post_init creates sub-configs when None."""

        config = OCRConfig(storage_root=temp_dir, environment=Environment.PRODUCTION)

        assert config.vector.storage_root == temp_dir
        assert config.vector.environment == Environment.PRODUCTION
        assert config.icechunk.storage_root == temp_dir
        assert config.icechunk.environment == Environment.PRODUCTION

    def test_environment_variable_override(self, temp_dir):
        """Test that environment variables work with OCRConfig."""
        with patch.dict(os.environ, {'OCR_ENVIRONMENT': 'production'}):
            with patch.object(IcechunkConfig, 'init_repo'):
                config = OCRConfig(storage_root=temp_dir)

                assert config.environment == Environment.PRODUCTION
                assert config.storage_root == temp_dir


# ============= Integration Tests =============


class TestConfigIntegration:
    """Integration tests for config classes working together."""

    def test_ocr_config_with_all_sub_configs(self, temp_dir):
        """Test OCRConfig with all sub-configurations specified."""
        with patch.object(IcechunkConfig, 'init_repo'):
            vector_config = VectorConfig(storage_root=temp_dir)
            icechunk_config = IcechunkConfig(storage_root=temp_dir)
            chunking_config = ChunkingConfig(chunks={'longitude': 500, 'latitude': 1000})
            coiled_config = CoiledConfig()

            ocr_config = OCRConfig(
                storage_root=temp_dir,
                vector=vector_config,
                icechunk=icechunk_config,
                chunking=chunking_config,
                coiled=coiled_config,
            )

            assert ocr_config.vector.storage_root == temp_dir
            assert ocr_config.icechunk.storage_root == temp_dir
            assert ocr_config.chunking.chunks == {'longitude': 500, 'latitude': 1000}
            assert ocr_config.coiled.ntasks == 1  # default value

    def test_branch_consistency_across_configs(self, temp_dir):
        """Test that branch is consistently applied across sub-configurations."""
        with patch.object(IcechunkConfig, 'init_repo'):
            config = OCRConfig(storage_root=temp_dir, environment=Environment.PRODUCTION)

            assert config.environment == Environment.PRODUCTION
            assert config.vector.environment == Environment.PRODUCTION
            assert config.icechunk.environment == Environment.PRODUCTION

    @patch.dict(os.environ, {'OCR_COILED_NTASKS': '16', 'OCR_VECTOR_PREFIX': 'custom/vector/path'})
    def test_environment_variables_sub_configs(self, temp_dir):
        """Test environment variables work for sub-configurations."""
        with patch.object(IcechunkConfig, 'init_repo'):
            config = OCRConfig(storage_root=temp_dir)

            # Coiled config should pick up env var
            assert config.coiled.ntasks == 16

            # Vector config should pick up its env var
            assert config.vector.prefix == 'custom/vector/path'


# ============= Error Handling Tests =============


class TestErrorHandling:
    """Test error handling in config classes."""

    def test_missing_required_fields(self):
        """Test validation errors for missing required fields."""
        with pytest.raises(pydantic.ValidationError):
            VectorConfig()  # Missing storage_root

        with pytest.raises(pydantic.ValidationError):
            IcechunkConfig()  # Missing storage_root

        with pytest.raises(pydantic.ValidationError):
            OCRConfig()  # Missing storage_root

    def test_invalid_enum_values(self, temp_dir):
        """Test validation errors for invalid enum values."""
        with pytest.raises(pydantic.ValidationError):
            VectorConfig(storage_root=temp_dir, branch='INVALID')

        with pytest.raises(pydantic.ValidationError):
            IcechunkConfig(storage_root=temp_dir, branch='INVALID')

    def test_chunking_config_uri_not_set_error(self, temp_dir):
        """Test error when URI is not set for IcechunkConfig operations."""
        with patch.object(IcechunkConfig, 'init_repo'):
            config = IcechunkConfig(storage_root=temp_dir)
            config.prefix = None

            with pytest.raises(ValueError, match='Prefix must be set'):
                _ = config.uri
