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
    license: str | None = None

    def to_xarray(
        self,
        *,
        is_icechunk: bool = False,
        xarray_open_kwargs: dict | None = None,
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

    def to_geopandas(
        self,
        query: str | None = None,
        geometry_column='geometry',
        crs: str = 'EPSG:4326',
        target_crs: str | None = None,
        **kwargs,
    ):
        """Convert query results to a GeoPandas GeoDataFrame.

        Parameters
        ----------
        query : str, optional
            SQL query to execute. If not provided, returns all data.
        geometry_column : str, default 'geometry'
            The name of the geometry column in the query result.
        crs : str, default 'EPSG:4326'
            The coordinate reference system to use for the geometries.
        target_crs : str, optional
            The target coordinate reference system to convert the geometries to.
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

        Example of converting buildings to GeoPandas GeoDataFrame - no need for ST_AsText():
        >>> buildings = catalog.get_dataset('conus-overture-buildings', 'v2025-03-19.1')
        >>> gdf = buildings.to_geopandas(\"\"\"
        ...     SELECT
        ...         id,
        ...         roof_material,
        ...         geometry
        ...     FROM read_parquet('{s3_path}')
        ...     WHERE roof_material = 'concrete'
        ... \"\"\")
        >>> gdf.head()

        """
        import geopandas as gpd
        from shapely import wkb, wkt

        if query is not None:
            # Only add the conversion if the geometry column doesn't already have a transformation
            if f'ST_AsText({geometry_column})' not in query:
                # Construct new query that preserves all columns but converts the geometry
                modified_query = query
                # Check if the query has a SELECT clause we can modify
                if 'SELECT' in query.upper() and 'FROM' in query.upper():
                    select_part, from_part = query.upper().split('FROM', 1)
                    # If the geometry column is directly selected (not already transformed)
                    if geometry_column.upper() in select_part:
                        original_query = query
                        modified_query = original_query.replace(
                            geometry_column, f'ST_AsText({geometry_column}) as {geometry_column}'
                        )

                query = modified_query

        # Get results

        result = self.query_geoparquet(query, **kwargs)
        df = result.df()

        # Check if geometry column exists
        if geometry_column not in df.columns:
            raise ValueError(f"Geometry column '{geometry_column}' not found in results")

        # Detect geometry format and convert appropriately
        sample = df[geometry_column].iloc[0] if not df.empty else None

        # Try different geometry conversion approaches
        try:
            # Try WKB format first (most common from ST_AsBinary)
            if isinstance(sample, bytes | bytearray):
                geometry_series = df[geometry_column].apply(
                    lambda g: wkb.loads(g) if g is not None else None
                )
            # Try WKT format (from ST_AsText)
            elif isinstance(sample, str) and (
                sample.startswith(('POINT', 'LINESTRING', 'POLYGON', 'MULTI'))
            ):
                geometry_series = df[geometry_column].apply(
                    lambda g: wkt.loads(g) if g is not None else None
                )
            # If we still have unrecognized format, try binary again with error handling
            else:
                try:
                    geometry_series = df[geometry_column].apply(
                        lambda g: wkb.loads(g) if g is not None else None
                    )
                except Exception:
                    # Final fallback - force to text and try WKT
                    # Create a temporary view to convert
                    duckdb.sql('CREATE OR REPLACE TEMP VIEW temp_geom AS SELECT * FROM df')
                    converted_df = duckdb.sql(
                        f'SELECT *, ST_AsText({geometry_column}) as geom_text FROM temp_geom'
                    ).df()
                    geometry_series = converted_df['geom_text'].apply(
                        lambda g: wkt.loads(g) if g is not None else None
                    )
        except Exception as e:
            raise ValueError(f'Failed to convert geometry column: {str(e)}')

        gdf = gpd.GeoDataFrame(df, geometry=geometry_series, crs=crs)

        if target_crs is not None:
            gdf = gdf.to_crs(target_crs)

        return gdf


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
    Dataset(
        name='alexandre-2016-digitized-buildings',
        description='Data from: Factors related to building loss due to wildfires in the conterminous United States',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/vector/alexandre-2016/digitized_buildings_2000_2010.parquet',
        data_format='geoparquet',
    ),
    Dataset(
        name='wildfire-communities',
        description='Wildfire Communities Dataset',
        bucket='carbonplan-risks',
        prefix='wildfirecommunities_short2023/short2023.zarr',
        data_format='zarr',
        version='v1',
    ),
    Dataset(
        name='cal-fire-damage-inspection',
        description='CAL FIRE Damage Inspection (DINS) dataset',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/vector/cal-fire-structures-destroyed/cal-fire-structures-destroyed.parquet',
        data_format='geoparquet',
    ),
    Dataset(
        name='era5-fire-weather-days',
        description='ERA5 Fire Weather Days',
        bucket='carbonplan-risks',
        prefix='era5/fire_weather_days_v2.zarr',
        data_format='zarr',
        version='v2',
    ),
]

catalog = Catalog(datasets=sorted(datasets, key=lambda x: x.name))
