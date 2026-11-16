"""Scott 2024 30 m CONUS grid split into regions: the parts fire score writes (`ocr.risks.fire.score`)."""

import functools
import json
from pathlib import Path

import numpy as np
import odc.geo.xr  # noqa
import pydantic
import pydantic_settings

from ocr import catalog


class ChunkingConfig(pydantic_settings.BaseSettings):
    chunks: dict | None = pydantic.Field(None, description='Chunk sizes for longitude and latitude')

    model_config = {
        'env_prefix': 'ocr_chunking_',
        'case_sensitive': False,
    }

    def model_post_init(self, __context):
        self.chunks = self.chunks or dict(zip(self.ds['CRPS'].dims, self.ds['CRPS'].data.chunksize))

    def __repr__(self):
        return self.extent.__repr__()

    @functools.cached_property
    def extent(self):
        from shapely.geometry import box

        return box(
            minx=float(self.ds.longitude.min()),
            maxx=float(self.ds.longitude.max()),
            miny=float(self.ds.latitude.min()),
            maxy=float(self.ds.latitude.max()),
        )

    @functools.cached_property
    def ds(self):
        dataset = (
            catalog.get_dataset('scott-et-al-2024-30m-4326').to_xarray().astype('float32')[['CRPS']]
        )
        dataset = dataset.odc.assign_crs('epsg:4326')
        return dataset

    @functools.cached_property
    def transform(self):
        return self.ds.odc.geobox.transform

    @functools.cached_property
    def chunk_info(self) -> dict:
        """Get information about the dataset's chunks"""
        y_chunks, x_chunks = self.ds['CRPS'].data.chunks
        y_starts = np.cumsum([0] + list(y_chunks[:-1]))
        x_starts = np.cumsum([0] + list(x_chunks[:-1]))

        return {
            'y_chunks': y_chunks,
            'x_chunks': x_chunks,
            'y_starts': y_starts,
            'x_starts': x_starts,
        }

    @functools.cached_property
    def valid_region_ids(self) -> list[str]:
        """Region ids (e.g. 'y1_x3') whose region has non-null Scott 2024 data."""
        # ponytail: committed file, nothing regenerates it; recompute if the Scott grid or chunks change.
        return json.loads((Path(__file__).parent / 'data' / 'valid_region_ids.json').read_text())

    def index_to_coords(self, x_idx: int, y_idx: int) -> tuple[float, float]:
        """Convert array indices to EPSG:4326 coordinates

        Parameters
        ----------
        x_idx : int
            Index along the x-dimension (longitude)
        y_idx : int
            Index along the y-dimension (latitude)

        Returns
        -------
        x, y : tuple[float, float]
            Corresponding EPSG:4326 coordinates (longitude, latitude)
        """
        x, y = self.transform * (x_idx, y_idx)
        return x, y

    def region_id_chunk_lookup(self, region_id: str) -> tuple:
        """given a region_id, ex: 'y5_x14, returns the corresponding chunk (5, 14)

        Parameters
        ----------
        region_id : str
            The region_id for chunk_id lookup.

        Returns
        -------
        index : tuple[int, int]
            The corresponding chunk (iy, ix) for the given region_id.
        """
        return self.get_chunk_mapping()[region_id]

    def chunk_id_to_slice(self, chunk_id: tuple) -> tuple:
        """
        Convert a chunk ID (iy, ix) to corresponding array slices

        Parameters
        ----------
        chunk_id : tuple
            The chunk identifier as a tuple (iy, ix) where:
            - iy is the index along y-dimension
            - ix is the index along x-dimension

        Returns
        -------
        chunk_slices : tuple[slice]
            A tuple of slices (y_slice, x_slice) to extract data for this chunk
        """
        iy, ix = chunk_id

        # Get chunk info
        chunk_info = self.chunk_info
        y_chunks = chunk_info['y_chunks']
        x_chunks = chunk_info['x_chunks']
        y_starts = chunk_info['y_starts']
        x_starts = chunk_info['x_starts']

        # Validate chunk indices
        if iy < 0 or iy >= len(y_chunks) or ix < 0 or ix >= len(x_chunks):
            raise ValueError(f'Invalid chunk ID: {chunk_id}. Out of bounds.')

        # Get start positions for this chunk
        y_start = y_starts[iy]
        x_start = x_starts[ix]

        # Get sizes for this chunk
        y_size = y_chunks[iy]
        x_size = x_chunks[ix]

        # Create and return the slices
        y_slice = slice(y_start, y_start + y_size)
        x_slice = slice(x_start, x_start + x_size)

        return (y_slice, x_slice)

    def region_id_to_latlon_slices(self, region_id: str) -> tuple:
        """
        Get latitude and longitude slices from region_id

        Returns (lat_slice, lon_slice) where lat_slice.start < lat_slice.stop
        and lon_slice.start < lon_slice.stop (lower-left origin, lat ascending).
        """
        chunk_id = self.region_id_chunk_lookup(region_id)
        # Get array index slices for this chunk (y, x)
        y_slice_idx, x_slice_idx = self.chunk_id_to_slice(chunk_id)

        # Convert corners to coordinates
        x_min, y_max = self.index_to_coords(x_slice_idx.start, y_slice_idx.start)  # upper-left
        x_max, y_min = self.index_to_coords(x_slice_idx.stop, y_slice_idx.stop)  # lower-right

        # Ensure slices are ascending (lower-left origin)
        lat_start, lat_stop = sorted((y_min, y_max))
        lon_start, lon_stop = sorted((x_min, x_max))

        lat_slice = slice(lat_start, lat_stop)
        lon_slice = slice(lon_start, lon_stop)

        return (lat_slice, lon_slice)

    def get_chunk_mapping(self) -> dict[str, tuple[int, int]]:
        """Returns a dict of region_ids and their corresponding chunk_indexes.

        Returns
        -------
        chunk_mapping : dict
            Dictionary with region IDs as keys and corresponding chunk indexes (iy, ix) as values
        """
        chunk_info = self.chunk_info
        y_starts = chunk_info['y_starts']
        x_starts = chunk_info['x_starts']

        chunk_mapping = {}
        for iy, y0 in enumerate(y_starts):
            for ix, x0 in enumerate(x_starts):
                chunk_mapping[f'y{iy}_x{ix}'] = (iy, ix)

        return chunk_mapping
