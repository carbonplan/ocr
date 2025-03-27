import typing

import duckdb
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

    def query_geoparquet(
        self,
        query: str | None = None,
        *,
        install_extensions: bool = True,
    ) -> 'duckdb.DuckDBPyRelation':
        """
        Query a geoparquet file using DuckDB.

        Parameters
        ----------
        query : str, optional
            SQL query to execute. If not provided, returns all data.
        install_extensions : bool, default True
            Whether to install and load the spatial and httpfs extensions.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Result of the DuckDB query.

        Raises
        ------
        ValueError
            If dataset is not in 'geoparquet' format.

        Example
        -------

        Example of querying buildings with a converted geometry column:

        >>> buildings = catalog.get_dataset('conus-overture-buildings', 'v2025-03-19.1')
        >>> result = buildings.query_geoparquet(\"\"\"
        ...     SELECT
        ...         id,
        ...         roof_material,
        ...         ST_AsText(geometry) as geometry
        ...     FROM read_parquet('{s3_path}')
        ...     WHERE roof_material = 'concrete'
        ... \"\"\")
        >>> # Then convert to GeoDataFrame
        >>> gdf = buildings.to_geopandas(\"\"\"
        ...     SELECT
        ...         id,
        ...         roof_material,
        ...         ST_AsText(geometry) as geometry
        ...     FROM read_parquet('{s3_path}')
        ...     WHERE roof_material = 'concrete'
        ... \"\"\")

        """
        if self.data_format != 'geoparquet':
            raise ValueError("Dataset must be in 'geoparquet' format to query with DuckDB.")

        import duckdb

        if install_extensions:
            duckdb.sql('INSTALL SPATIAL; LOAD SPATIAL; INSTALL httpfs; LOAD httpfs')

        s3_path = f's3://{self.bucket}/{self.prefix}'

        if query is None:
            return duckdb.sql(f"SELECT * FROM read_parquet('{s3_path}')")
        else:
            # Replace placeholder in query if present
            if '{s3_path}' in query:
                query = query.format(s3_path=s3_path)
            return duckdb.sql(query)

    def to_geopandas(self, query: str | None = None, geometry_column='geometry', **kwargs):
        """Convert query results to a GeoPandas GeoDataFrame.

        Parameters
        ----------
        query : str, optional
            SQL query to execute. If not provided, returns all data.
        geometry_column : str, default 'geometry'
            The name of the geometry column in the query result.
        **kwargs : dict
            Additional keyword arguments passed to `query_geoparquet`.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoPandas GeoDataFrame containing the queried data with geometries.

        Raises
        ------
        ValueError
            If dataset is not in 'geoparquet' format or if the geometry column is not found.

        Example
        -------

        Example of converting buildings with a converted geometry column to GeoPandas GeoDataFrame:
        >>> buildings = catalog.get_dataset('conus-overture-buildings', 'v2025-03-19.1')
        >>> gdf = buildings.to_geopandas(\"\"\"
        ...     SELECT
        ...         id,
        ...         roof_material,
        ...         ST_AsText(geometry) as geometry
        ...     FROM read_parquet('{s3_path}')
        ...     WHERE roof_material = 'concrete'
        ... \"\"\")
        >>> gdf.head()

        """
        import geopandas as gpd
        from shapely import wkt

        result = self.query_geoparquet(query, **kwargs)
        df = result.df()
        return gpd.GeoDataFrame(df, geometry=df[geometry_column].apply(wkt.loads), crs='EPSG:4326')


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
