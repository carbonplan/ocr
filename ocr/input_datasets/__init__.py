"""Input dataset processing infrastructure for OCR.

This module provides base classes, utilities, and processors for downloading,
processing, and uploading input datasets to S3/Icechunk storage.
"""

from ocr.input_datasets.base import BaseDatasetProcessor, InputDatasetConfig
from ocr.input_datasets.storage import IcechunkWriter, S3Uploader

__all__ = [
    'BaseDatasetProcessor',
    'InputDatasetConfig',
    'IcechunkWriter',
    'S3Uploader',
]
