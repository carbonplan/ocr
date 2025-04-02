# flake8: noqa
import importlib.metadata

# get the version of the package
__version__ = importlib.metadata.version('ocr')
from ocr.dep_versions import show_versions
from ocr.datasets import catalog
from ocr.cal_fire_dins import load_structures_destroyed
