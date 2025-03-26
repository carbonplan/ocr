import pydantic
import typing
import xarray as xr 
import icechunk as ic

class Dataset(pydantic.BaseModel):
    """
    Base class for datasets.
    """

    name: str
    description: str
    bucket: str
    prefix: str
    data_format: typing.Literal["parquet", "zarr", "csv", "shapefile"]

  
    def to_xarray(self, *, is_icechunk: bool = False, xarray_open_kwargs: dict | None = None, xarray_storage_options: dict | None = None) -> xr.Dataset:
        """
        Convert the dataset to an xarray.Dataset.
        """
        if self.data_format != "zarr":
            raise ValueError("Dataset must be in 'zarr' format to convert to xarray.")
        
        if xarray_open_kwargs is None:
            xarray_open_kwargs = {}

        
        if xarray_storage_options is None:
            xarray_storage_options = {}
        
        if is_icechunk:
            storage = ic.s3_storage(bucket=self.bucket, prefix=self.prefix)
            repo = ic.Repository(storage=storage)
            session = repo.readonly_session('main')
            xarray_open_kwargs = {**xarray_open_kwargs, **{'consolidated': False, 'engine': 'zarr'}}
            ds = xr.open_dataset(session.store, **xarray_open_kwargs)
        else:
            ds = xr.open_dataset(f"s3://{self.bucket}/{self.prefix}", **xarray_open_kwargs, storage_options=xarray_storage_options)
        return ds
            
        
     

        
        
        


    @classmethod
    def to_geopandas(cls):
        """
        Convert the dataset to a geopandas.GeoDataFrame.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def to_pandas(cls):
        """
        Convert the dataset to a pandas.DataFrame.
        """
        raise NotImplementedError("Subclasses must implement this method.")




