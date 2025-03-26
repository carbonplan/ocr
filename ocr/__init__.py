# flake8: noqa
import importlib.metadata

# get the version of the package
__version__ = importlib.metadata.version('ocr')
from ocr.dep_versions import show_versions
from ocr.datasets import catalog
