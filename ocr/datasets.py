import typing

import icechunk as ic
import pydantic
import xarray as xr


class Dataset(pydantic.BaseModel):
    """
    Base class for datasets.
    """

    name: str
    description: str
    bucket: str
    prefix: str
    data_format: typing.Literal['geoparquet', 'zarr']
    version: str = 'v1'

    def to_xarray(
        self,
        *,
        is_icechunk: bool = False,
        xarray_open_kwargs: dict,
        xarray_storage_options: dict | None = None,
    ) -> xr.Dataset:
        """
        Convert the dataset to an xarray.Dataset.
        """
        if self.data_format != 'zarr':
            raise ValueError("Dataset must be in 'zarr' format to convert to xarray.")

        if xarray_open_kwargs is None:
            xarray_open_kwargs = {}

        if xarray_storage_options is None:
            xarray_storage_options = {}

        if is_icechunk:
            storage = ic.s3_storage(bucket=self.bucket, prefix=self.prefix)
            repo = ic.Repository.open(storage=storage)
            session = repo.readonly_session('main')
            xarray_open_kwargs = {**xarray_open_kwargs, **{'consolidated': False, 'engine': 'zarr'}}
            ds = xr.open_dataset(session.store, **xarray_open_kwargs)
        else:
            ds = xr.open_dataset(
                f's3://{self.bucket}/{self.prefix}',
                **xarray_open_kwargs,
                storage_options=xarray_storage_options,
            )
        return ds


class Catalog(pydantic.BaseModel):
    """
    Base class for datasets catalog.
    """

    datasets: list[Dataset]

    def get_dataset(self, name: str, version: str) -> Dataset | None:
        """
        Get a dataset by name.
        """
        for dataset in self.datasets:
            if dataset.name == name and dataset.version == version:
                return dataset
        raise KeyError(f"Dataset '{name}' with version '{version}' not found in the catalog.")

    def __iter__(self):
        return iter(self.datasets)

    def __str__(self) -> str:
        """
        Return a string representation of the catalog.
        """
        return self.__repr__()

    def __repr__(self) -> str:
        """
        Return a string representation of the catalog.
        """
        try:
            from io import StringIO

            from rich.console import Console
            from rich.table import Table

            output = StringIO()
            console = Console(file=output)

            # Create the table
            table = Table(
                title=f'📊 OCR Dataset Catalog ({len(self.datasets)} datasets)',
                show_lines=True,
                expand=True,
            )

            # Add columns
            table.add_column('Name', style='cyan bold', ratio=3, overflow='fold')
            table.add_column('Description', style='green', ratio=4, overflow='fold')
            table.add_column('Format', style='magenta', ratio=1, overflow='fold')
            table.add_column('Version', style='blue', ratio=2, overflow='fold')
            table.add_column('Storage Location', style='yellow', ratio=5, overflow='fold')

            # Add rows
            for ds in self.datasets:
                table.add_row(
                    ds.name,
                    ds.description,
                    f'{ds.data_format}',
                    ds.version,
                    f's3://{ds.bucket}/{ds.prefix}',
                )

            console.print(table)
            return output.getvalue()
        except ImportError:
            # Fallback if rich is not available
            result = [f'📊 OCR Dataset Catalog ({len(self.datasets)} datasets)']
            for ds in self.datasets:
                result.append(f'- {ds.name}: {ds.description} [{ds.data_format}]')
            return '\n'.join(result)


datasets = [
    Dataset(
        name='2011-climate-run',
        description='USFS 2011 Climate Run',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/USFS/2011ClimateRun_Icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='2047-climate-run',
        description='USFS 2047 Climate Run',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/USFS/2047ClimateRun_Icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='conus-overture-addresses',
        description='CONUS Overture Addresses',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/vector/CONUS_overture_addresses_2025-03-19.1.parquet',
        data_format='geoparquet',
        version='v2025-03-19.1',
    ),
    Dataset(
        name='conus-overture-buildings',
        description='CONUS Overture Buildings',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/vector/CONUS_overture_buildings_2025-03-19.1.parquet',
        data_format='geoparquet',
        version='v2025-03-19.1',
    ),
]

catalog = Catalog(datasets=datasets)
