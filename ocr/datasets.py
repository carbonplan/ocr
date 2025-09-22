import typing

import duckdb
import geopandas as gpd
import icechunk as ic
import pydantic
import xarray as xr
from shapely import wkt


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
        is_icechunk: bool | None = None,
        xarray_open_kwargs: dict | None = None,
        xarray_storage_options: dict | None = None,
    ) -> xr.Dataset:
        """
        Convert the dataset to an xarray.Dataset.

        Parameters
        ----------
        is_icechunk : bool | None, default None
            Whether to use icechunk to access the data.
            - If True: only try using icechunk
            - If None: try icechunk first, fall back to direct S3 access if it fails
            - If False: only use direct S3 access
        xarray_open_kwargs : dict, optional
            Additional keyword arguments to pass to xarray.open_dataset.
        xarray_storage_options : dict, optional
            Storage options for S3 access when not using icechunk.

        Returns
        -------
        xr.Dataset
            The opened dataset.

        Raises
        ------
        ValueError
            If the dataset is not in 'zarr' format.
        FileNotFoundError
            If the dataset cannot be found or accessed.
        """
        if self.data_format != 'zarr':
            raise ValueError("Dataset must be in 'zarr' format to convert to xarray.")

        xarray_open_kwargs = xarray_open_kwargs or {}
        xarray_storage_options = xarray_storage_options or {}

        # Try icechunk if is_icechunk is True or None
        if is_icechunk is not False:
            try:
                storage = ic.s3_storage(bucket=self.bucket, prefix=self.prefix)
                repo = ic.Repository.open(storage=storage)
                session = repo.readonly_session('main')

                icechunk_kwargs = {
                    'consolidated': False,
                    'engine': 'zarr',
                    'chunks': {},
                    **xarray_open_kwargs,
                }

                ds = xr.open_dataset(session.store, **icechunk_kwargs)
                return ds
            except Exception as exc:
                # If is_icechunk=True but icechunk failed, raise the error
                if is_icechunk is True:
                    raise FileNotFoundError(
                        f"Failed to open icechunk repository: '{self.bucket}/{self.prefix}'"
                    ) from exc
                # Otherwise, if is_icechunk=None, we'll try the fallback method

        # Direct S3 access (either is_icechunk=False or icechunk failed with is_icechunk=None)
        try:
            direct_s3_kwargs = {'engine': 'zarr', 'chunks': {}, **xarray_open_kwargs}

            ds = xr.open_dataset(
                f's3://{self.bucket}/{self.prefix}',
                **direct_s3_kwargs,
                storage_options=xarray_storage_options,
            )
            return ds
        except Exception as exc:
            raise FileNotFoundError(
                f"No such file or directory: 's3://{self.bucket}/{self.prefix}'"
            ) from exc

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
        ...         geometry
        ...     FROM read_parquet('{s3_path}')
        ...     WHERE roof_material = 'concrete'
        ... \"\"\")
        >>> # Then convert to GeoDataFrame
        >>> gdf = buildings.to_geopandas(\"\"\"
        ...     SELECT
        ...         id,
        ...         roof_material,
        ...         geometry
        ...     FROM read_parquet('{s3_path}')
        ...     WHERE roof_material = 'concrete'
        ... \"\"\")

        """
        if self.data_format != 'geoparquet':
            raise ValueError("Dataset must be in 'geoparquet' format to query with DuckDB.")

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
    ) -> gpd.GeoDataFrame:
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

        if query is not None:
            # Only add the conversion if the geometry column doesn't already have a transformation
            if f'ST_AsText({geometry_column})' not in query:
                # Construct new query that preserves all columns but converts the geometry
                modified_query = query
                # Check if the query has a SELECT clause we can modify
                if 'SELECT' in query.upper() and 'FROM' in query.upper():
                    select_part, _ = query.upper().split('FROM', 1)

                    # case 1: SELECT * query
                    if 'SELECT *' in query.upper():
                        # get the column names to build an explicit query
                        columns_query = "SELECT * FROM read_parquet('{s3_path}') LIMIT 0"
                        columns_result = self.query_geoparquet(columns_query, **kwargs)
                        columns_df = columns_result.df()

                        # Create new query with explicit columns
                        col_list = []
                        for col in columns_df.columns:
                            if col.lower() == geometry_column.lower():
                                col_list.append(f'ST_AsText({col}) as {col}')
                            else:
                                col_list.append(col)
                        # Replace SELECT * with explicit column list
                        modified_query = query.replace('*', ', '.join(col_list), 1)

                    # If the geometry column is directly selected (not already transformed)
                    elif geometry_column.upper() in select_part:
                        original_query = query
                        modified_query = original_query.replace(
                            geometry_column,
                            f'ST_AsText({geometry_column}) as {geometry_column}',
                        )
                query = modified_query

        result = self.query_geoparquet(query, **kwargs)
        df = result.df()

        if geometry_column not in df.columns:
            raise ValueError(f"Geometry column '{geometry_column}' not found in results")

        try:
            geometry_series = df[geometry_column].apply(
                lambda g: wkt.loads(g) if g is not None else None
            )

            gdf = gpd.GeoDataFrame(df, geometry=geometry_series, crs=crs)

            if target_crs is not None:
                gdf = gdf.to_crs(target_crs)

            return gdf
        except Exception as e:
            raise ValueError(f'Failed to convert geometry column: {geometry_column}') from e


class Catalog(pydantic.BaseModel):
    """
    Base class for datasets catalog.
    """

    datasets: list[Dataset]

    def get_dataset(
        self,
        name: str,
        version: str | None = None,
        *,
        case_sensitive: bool = True,
        latest: bool = False,
    ) -> Dataset:
        """
        Get a dataset by name and optionally version.

        Parameters
        ----------
        name : str
            Name of the dataset to retrieve
        version : str, optional
            Specific version of the dataset. If not provided, returns the dataset
            if only one version exists, or raises an error if multiple versions exist,
            unless get_latest=True.
        case_sensitive : bool, default True
            Whether to match dataset names case-sensitively
        latest : bool, default False
            If True and version=None, returns the latest version instead of raising
            an error when multiple versions exist


        Returns
        -------
        Dataset
            The matched dataset

        Raises
        ------
        ValueError
            If multiple versions exist and version is not specified (and latest=False)
        KeyError
            If no matching dataset is found

        Examples
        --------
        >>> # Get a dataset with a specific version
        >>> catalog.get_dataset('conus-overture-buildings', 'v2025-03-19.1')
        >>>
        >>> # Get latest version of a dataset
        >>> catalog.get_dataset('conus-overture-buildings', get_latest=True)
        """

        found_datasets = []
        name_matches = []

        for dataset in self.datasets:
            dataset_name = dataset.name if case_sensitive else dataset.name.lower()
            search_name = name if case_sensitive else name.lower()

            if dataset_name == search_name:
                name_matches.append(dataset.name)
                if version is None or dataset.version == version:
                    found_datasets.append(dataset)

        if version is None:
            if len(found_datasets) == 1:
                return found_datasets[0]
            elif len(found_datasets) > 1:
                if latest:
                    try:
                        return sorted(found_datasets, key=lambda x: x.version, reverse=True)[0]
                    except Exception as e:
                        found_versions = {dataset.version for dataset in found_datasets}
                        raise ValueError(
                            f'Could not determine the latest version from {found_versions}. '
                            f'Please specify a version explicitly.'
                        ) from e
                else:
                    found_versions = {dataset.version for dataset in found_datasets}
                    raise ValueError(
                        f"Multiple versions found for dataset '{name}'. "
                        f'Please specify a version: {sorted(found_versions)} '
                        f'or use get_latest=True to automatically select the latest version.'
                    )

        if found_datasets:
            return found_datasets[0]

        if name_matches:
            # We found the name but not the specific version
            found_versions = {
                dataset.version
                for dataset in self.datasets
                if (
                    dataset.name == name if case_sensitive else dataset.name.lower() == name.lower()
                )
            }
            raise KeyError(
                f"Dataset '{name}' exists, but version '{version}' was not found. "
                f'Available versions: {sorted(found_versions)}'
            )

        raise KeyError(f"Dataset '{name}' not found in the catalog.")

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
        name='2011-climate-run-30m-4326',
        description='USFS 2011 Climate Run',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/USFS/2011_climate_run_30m_4326_chunked_icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='2047-climate-run-30m-4326',
        description='USFS 2047 Climate Run',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/USFS/2047_climate_run_30m_4326_chunked_icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='RDS-2016-0032-3',
        description='Spatial datasets of probabilistic wildfire risk components for the United States (270m)',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/USFS/RDS-2016-0034-3-epsg_4326.icechunk',
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
        prefix='input/fire-risk/vector/CONUS_overture_buildings_2025-06-25.0.parquet',
        data_format='geoparquet',
        version='v2025-03-19.1',
    ),
    Dataset(
        name='conus-overture-buildings-5070',
        description='CONUS Overture Buildings in EPSG 5070. Columns are: bbox, bbox_5070, geometry, geometry_5070',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/vector/CONUS_overture_buildings_5070_2025-03-19.1.parquet',
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
        name='USFS-wildfire-risk-communities',
        description='Wildfire Risk to Communities - RDS-2020-0016-2',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/USFS/RDS-2022-0016-02_all_vars_merge_icechunk/',
        data_format='zarr',
    ),
    Dataset(
        name='USFS-wildfire-risk-communities-4326',
        description='Wildfire Risk to Communities - RDS-2020-0016-2',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/USFS/RDS-2022-0016-02_EPSG_4326_icechunk_all_vars',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-hourly-Q2',
        description='Q2 variable from CONUS404 hourly data in Icechunk format',
        bucket='carbonplan-ocr',
        prefix='input/conus404-hourly-icechunk/Q2',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-hourly-TD2',
        description='TD2 variable from CONUS404 hourly data in Icechunk format',
        bucket='carbonplan-ocr',
        prefix='input/conus404-hourly-icechunk/TD2',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-hourly-PSFC',
        description='PSFC variable from CONUS404 hourly data in Icechunk format',
        bucket='carbonplan-ocr',
        prefix='input/conus404-hourly-icechunk/PSFC',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-hourly-T2',
        description='T2 variable from CONUS404 hourly data in Icechunk format',
        bucket='carbonplan-ocr',
        prefix='input/conus404-hourly-icechunk/T2',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-hourly-V10',
        description='V10 variable from CONUS404 hourly data in Icechunk format',
        bucket='carbonplan-ocr',
        prefix='input/conus404-hourly-icechunk/V10',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-hourly-U10',
        description='U10 variable from CONUS404 hourly data in Icechunk format',
        bucket='carbonplan-ocr',
        prefix='input/conus404-hourly-icechunk/U10',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-fire-weather-wind-mode-hurs15-wind35',
        description='Modal wind direction (0-7 cardinal) during fire-weather hours (RH<15%, gust-like wind>35 mph) on native CONUS404 grid',
        bucket='carbonplan-ocr',
        prefix='input/conus404-wind-direction-modes/fire_weather_wind_mode-hurs15_wind35.zarr',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-fire-weather-wind-mode-hurs15-wind35-reprojected',
        description='Modal wind direction (0-7 cardinal) during fire-weather hours (RH<15%, gust-like wind>35 mph) reprojected to USFS wildfire risk geobox (EPSG:4326)',
        bucket='carbonplan-ocr',
        prefix='input/conus404-wind-direction-modes/fire_weather_wind_mode-hurs15_wind35-reprojected.zarr',
        data_format='zarr',
    ),
    Dataset(
        name='us-census-tracts',
        description='US Census Tracts',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/vector/aggregated_regions/tracts/tracts.parquet',
        data_format='geoparquet',
    ),
    Dataset(
        name='us-census-counties',
        description='US Census Counties',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/vector/aggregated_regions/counties/counties.parquet',
        data_format='geoparquet',
    ),
    # CONUS404 Fosberg Fire Weather Index (FFWI) datasets
    Dataset(
        name='conus404-ffwi',
        description='Fosberg Fire Weather Index (FFWI) on CONUS404 native grid',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index.icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-ffwi-p95',
        description='FFWI p95 (95th percentile) on CONUS404 native grid',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index_p95.icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-ffwi-p95-mode',
        description='Modal wind direction during FFWI p95 conditions on CONUS404 native grid',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index_p95_mode.icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-ffwi-p95-wind-direction-distribution',
        description='Wind direction distribution during FFWI p95 conditions on CONUS404 native grid',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index_p95_wind_direction_distribution.icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-ffwi-p99',
        description='FFWI p99 (99th percentile) on CONUS404 native grid',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index_p99.icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-ffwi-p99-mode',
        description='Modal wind direction during FFWI p99 conditions on CONUS404 native grid',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index_p99_mode.icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-ffwi-p99-wind-direction-distribution',
        description='Wind direction distribution during FFWI p99 conditions on CONUS404 native grid',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index_p99_wind_direction_distribution.icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-ffwi-winds',
        description='Wind variables associated with FFWI computations on CONUS404 native grid',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/conus404-ffwi/winds.icechunk',
        data_format='zarr',
    ),
    Dataset(
        name='conus404-ffwi-p99-mode-reprojected',
        description='Modal wind direction during FFWI p99 conditions reprojected to USFS wildfire risk geobox (EPSG:4326)',
        bucket='carbonplan-ocr',
        prefix='input/fire-risk/tensor/conus404-ffwi/fosberg-fire-weather-index_p99_mode_reprojected.icechunk',
        data_format='zarr',
    ),
]


catalog = Catalog(datasets=sorted(datasets, key=lambda x: x.name))
