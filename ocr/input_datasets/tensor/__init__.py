"""Tensor (raster/gridded) input dataset processors."""

from ocr.input_datasets.tensor.calfire_fhsz import CalfireFHSZProcessor
from ocr.input_datasets.tensor.conus404 import Conus404FFWIProcessor, Conus404SubsetProcessor
from ocr.input_datasets.tensor.usfs_dillon_2023 import Dillon2023Processor
from ocr.input_datasets.tensor.usfs_riley_2025 import RileyEtAl2025Processor
from ocr.input_datasets.tensor.usfs_scott_2024 import ScottEtAl2024Processor

__all__ = [
    'CalfireFHSZProcessor',
    'Conus404FFWIProcessor',
    'Conus404SubsetProcessor',
    'Dillon2023Processor',
    'RileyEtAl2025Processor',
    'ScottEtAl2024Processor',
]
