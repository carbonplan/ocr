import importlib.metadata

# get the version of the package
__version__ = importlib.metadata.version('ocr')


def __getattr__(name: str):
    # Fire's catalog loads lazily so the shared modules don't import fire. TODO: drop with the fire refactor.
    if name == 'catalog':
        from ocr.risks.fire.datasets import catalog

        return catalog
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
