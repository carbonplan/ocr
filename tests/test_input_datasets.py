"""Tests for input dataset processing infrastructure."""

import tempfile
from pathlib import Path

from ocr.input_datasets import InputDatasetConfig
from ocr.input_datasets.storage import IcechunkWriter, S3Uploader
from ocr.input_datasets.tensor.usfs_scott_2024 import ScottEtAl2024Processor


class TestInputDatasetConfig:
    """Test InputDatasetConfig."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = InputDatasetConfig()
        assert config.s3_bucket == 'carbonplan-ocr'
        assert config.s3_region == 'us-west-2'
        assert config.base_prefix == 'input/fire-risk'
        assert config.chunk_size == 8192
        assert config.debug is False

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = InputDatasetConfig(s3_bucket='test-bucket', s3_region='us-east-1', debug=True)
        assert config.s3_bucket == 'test-bucket'
        assert config.s3_region == 'us-east-1'
        assert config.debug is True


class TestIcechunkWriter:
    """Test IcechunkWriter utility."""

    def test_init(self):
        """Test IcechunkWriter initialization."""
        writer = IcechunkWriter(
            bucket='test-bucket', prefix='test/prefix', region='us-west-2', dry_run=True
        )
        assert writer.bucket == 'test-bucket'
        assert writer.prefix == 'test/prefix'
        assert writer.region == 'us-west-2'
        assert writer.dry_run is True

    def test_dry_run_write(self):
        """Test dry run write operation."""
        import xarray as xr

        writer = IcechunkWriter(
            bucket='test-bucket', prefix='test/prefix', region='us-west-2', dry_run=True
        )

        # Create a minimal dataset
        ds = xr.Dataset({'var': (['x', 'y'], [[1, 2], [3, 4]])})

        # Should return without error in dry run mode
        snapshot_id = writer.write(ds, 'test commit')
        assert snapshot_id == 'dry-run-snapshot-id'


class TestS3Uploader:
    """Test S3Uploader utility."""

    def test_init(self):
        """Test S3Uploader initialization."""
        uploader = S3Uploader(bucket='test-bucket', region='us-west-2', dry_run=True)
        assert uploader.bucket == 'test-bucket'
        assert uploader.region == 'us-west-2'
        assert uploader.dry_run is True

    def test_dry_run_upload(self):
        """Test dry run upload operation."""
        uploader = S3Uploader(bucket='test-bucket', region='us-west-2', dry_run=True)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            tmp.write(b'test content')
            tmp_path = Path(tmp.name)

        try:
            # Should return S3 URI without error in dry run mode
            uri = uploader.upload_file(tmp_path, 'test/key.txt')
            assert uri == 's3://test-bucket/test/key.txt'
        finally:
            tmp_path.unlink()


class TestScottEtAl2024Processor:
    """Test ScottEtAl2024Processor."""

    def test_init(self):
        """Test processor initialization."""
        processor = ScottEtAl2024Processor(dry_run=True)
        assert processor.dataset_name == 'scott-et-al-2024'
        assert processor.dataset_type == 'tensor'
        assert processor.version == '2024-V2'
        assert processor.rds_id == 'RDS-2020-0016-02'
        assert processor.dry_run is True

    def test_variables_defined(self):
        """Test that all 8 variables are defined."""
        processor = ScottEtAl2024Processor(dry_run=True)
        expected_vars = ['BP', 'CRPS', 'CFL', 'Exposure', 'FLEP4', 'FLEP8', 'RPS', 'WHP']
        assert list(processor.VARIABLES.keys()) == expected_vars
        # Verify all have URLs
        for var, url in processor.VARIABLES.items():
            assert url.startswith('https://usfs-public.box.com/')

    def test_s3_prefixes(self):
        """Test S3 prefix generation."""
        processor = ScottEtAl2024Processor(dry_run=True)
        assert processor.s3_tiff_prefix == 'input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif'
        assert (
            processor.s3_icechunk_prefix
            == 'input/fire-risk/tensor/USFS/RDS-2020-0016-02_all_vars_merge_icechunk'
        )
        assert (
            processor.s3_icechunk_4326_prefix
            == 'input/fire-risk/tensor/USFS/scott-et-al-2024-30m-4326.icechunk'
        )

    def test_temp_dir_creation(self):
        """Test temporary directory creation."""
        processor = ScottEtAl2024Processor(dry_run=True)
        temp_dir = processor.temp_dir
        assert temp_dir.name == 'scott-et-al-2024'
        assert temp_dir.exists()
        # Cleanup
        processor.cleanup_temp()
        assert not temp_dir.exists()

    def test_download_file_dry_run(self):
        """Test download in dry run mode."""
        processor = ScottEtAl2024Processor(dry_run=True)
        output_path = processor.temp_dir / 'test.zip'
        result = processor.download_file('https://example.com/test.zip', output_path)
        assert result == output_path
        processor.cleanup_temp()
