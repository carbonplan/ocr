import importlib
import sys


def show_versions(file=sys.stdout):
    """print the versions of dependencies.
       Adapted from xarray/util/print_versions.py

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """

    deps = [
        ('xarray', lambda mod: mod.__version__),
        ('geopandas', lambda mod: mod.__version__),
        ('rioxarray', lambda mod: mod.__version__),
        ('windrose', lambda mod: mod.__version__),
        ('matplotlib', lambda mod: mod.__version__),
        ('cv2', lambda mod: mod.__version__),
        ('s3fs', lambda mod: mod.__version__),
        ('scipy', lambda mod: mod.__version__),
        ('dask', lambda mod: mod.__version__),
        ('zarr', lambda mod: mod.__version__),
        ('icechunk', lambda mod: mod.__version__),
        ('shapely', lambda mod: mod.__version__),
        ('coiled', lambda mod: mod.__version__),
    ]

    deps_blob = []
    for modname, ver_f in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
        except Exception:
            deps_blob.append((modname, None))
        else:
            try:
                ver = ver_f(mod)
                deps_blob.append((modname, ver))
            except Exception:
                deps_blob.append((modname, 'installed'))

    print('\nINSTALLED VERSIONS', file=file)
    print('------------------', file=file)

    print('', file=file)
    for k, stat in sorted(deps_blob):
        print(f'{k}: {stat}', file=file)
