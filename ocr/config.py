import functools
import random
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import dotenv
import icechunk
import numpy as np
import odc.geo.xr  # noqa
import pydantic
import pydantic_settings
import xarray as xr
from pydantic_extra_types.semantic_version import SemanticVersion
from upath import UPath

from ocr import catalog
from ocr.console import console
from ocr.types import Environment


class CoiledConfig(pydantic_settings.BaseSettings):
    tag: dict[str, str] = pydantic.Field({'Project': 'OCR'})
    forward_aws_credentials: bool = pydantic.Field(
        False, description='Whether to forward AWS credentials to the worker nodes'
    )
    spot_policy: typing.Literal['on-demand', 'spot', 'spot_with_fallback'] = pydantic.Field(
        'spot_with_fallback',
        description='Spot instance policy for Coiled cluster. See Coiled docs for details.',
    )
    region: str = pydantic.Field('us-west-2', description='AWS region to use for the worker nodes')
    ntasks: pydantic.PositiveInt = pydantic.Field(
        1, description='Number of tasks to run in parallel'
    )
    vm_type: str = pydantic.Field('m8g.2xlarge', description='VM type to use for the worker nodes')
    scheduler_vm_type: str = pydantic.Field(
        'm8g.2xlarge', description='VM type to use for the scheduler node'
    )

    model_config = {
        'env_prefix': 'ocr_coiled_',
        'case_sensitive': False,
    }


class ChunkingConfig(pydantic_settings.BaseSettings):
    chunks: dict | None = pydantic.Field(None, description='Chunk sizes for longitude and latitude')
    debug: bool = pydantic.Field(False, description='Enable debugging mode')

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
    def extent_as_tuple(self):
        bounds = self.extent.bounds
        return (bounds[0], bounds[2], bounds[1], bounds[3])

    @functools.cached_property
    def ds(self):
        dataset = (
            catalog.get_dataset('USFS-wildfire-risk-communities-4326')
            .to_xarray()
            .astype('float32')[['CRPS']]
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
    def valid_region_ids(self) -> list:
        # generated and saved from generate_valid_region_ids()
        return [
            'y1_x3',
            'y1_x4',
            'y1_x5',
            'y1_x6',
            'y1_x7',
            'y1_x8',
            'y1_x9',
            'y1_x10',
            'y1_x11',
            'y1_x12',
            'y1_x13',
            'y1_x14',
            'y1_x15',
            'y1_x16',
            'y1_x17',
            'y1_x18',
            'y1_x19',
            'y1_x20',
            'y1_x21',
            'y1_x22',
            'y1_x23',
            'y1_x24',
            'y2_x2',
            'y2_x3',
            'y2_x4',
            'y2_x5',
            'y2_x6',
            'y2_x7',
            'y2_x8',
            'y2_x9',
            'y2_x10',
            'y2_x11',
            'y2_x12',
            'y2_x13',
            'y2_x14',
            'y2_x15',
            'y2_x16',
            'y2_x17',
            'y2_x18',
            'y2_x19',
            'y2_x20',
            'y2_x21',
            'y2_x22',
            'y2_x23',
            'y2_x24',
            'y2_x25',
            'y2_x26',
            'y2_x27',
            'y2_x28',
            'y2_x29',
            'y2_x30',
            'y2_x31',
            'y2_x42',
            'y2_x43',
            'y3_x2',
            'y3_x3',
            'y3_x4',
            'y3_x5',
            'y3_x6',
            'y3_x7',
            'y3_x8',
            'y3_x9',
            'y3_x10',
            'y3_x11',
            'y3_x12',
            'y3_x13',
            'y3_x14',
            'y3_x15',
            'y3_x16',
            'y3_x17',
            'y3_x18',
            'y3_x19',
            'y3_x20',
            'y3_x21',
            'y3_x22',
            'y3_x23',
            'y3_x24',
            'y3_x25',
            'y3_x26',
            'y3_x27',
            'y3_x28',
            'y3_x29',
            'y3_x30',
            'y3_x31',
            'y3_x32',
            'y3_x33',
            'y3_x41',
            'y3_x42',
            'y3_x43',
            'y3_x44',
            'y4_x2',
            'y4_x3',
            'y4_x4',
            'y4_x5',
            'y4_x6',
            'y4_x7',
            'y4_x8',
            'y4_x9',
            'y4_x10',
            'y4_x11',
            'y4_x12',
            'y4_x13',
            'y4_x14',
            'y4_x15',
            'y4_x16',
            'y4_x17',
            'y4_x18',
            'y4_x19',
            'y4_x20',
            'y4_x21',
            'y4_x22',
            'y4_x23',
            'y4_x24',
            'y4_x25',
            'y4_x26',
            'y4_x27',
            'y4_x28',
            'y4_x29',
            'y4_x30',
            'y4_x31',
            'y4_x32',
            'y4_x33',
            'y4_x35',
            'y4_x36',
            'y4_x37',
            'y4_x38',
            'y4_x39',
            'y4_x40',
            'y4_x41',
            'y4_x42',
            'y4_x43',
            'y4_x44',
            'y5_x2',
            'y5_x3',
            'y5_x4',
            'y5_x5',
            'y5_x6',
            'y5_x7',
            'y5_x8',
            'y5_x9',
            'y5_x10',
            'y5_x11',
            'y5_x12',
            'y5_x13',
            'y5_x14',
            'y5_x15',
            'y5_x16',
            'y5_x17',
            'y5_x18',
            'y5_x19',
            'y5_x20',
            'y5_x21',
            'y5_x22',
            'y5_x23',
            'y5_x24',
            'y5_x25',
            'y5_x26',
            'y5_x27',
            'y5_x28',
            'y5_x29',
            'y5_x30',
            'y5_x31',
            'y5_x32',
            'y5_x33',
            'y5_x34',
            'y5_x35',
            'y5_x36',
            'y5_x37',
            'y5_x38',
            'y5_x39',
            'y5_x40',
            'y5_x41',
            'y5_x42',
            'y6_x2',
            'y6_x3',
            'y6_x4',
            'y6_x5',
            'y6_x6',
            'y6_x7',
            'y6_x8',
            'y6_x9',
            'y6_x10',
            'y6_x11',
            'y6_x12',
            'y6_x13',
            'y6_x14',
            'y6_x15',
            'y6_x16',
            'y6_x17',
            'y6_x18',
            'y6_x19',
            'y6_x20',
            'y6_x21',
            'y6_x22',
            'y6_x23',
            'y6_x24',
            'y6_x25',
            'y6_x26',
            'y6_x27',
            'y6_x28',
            'y6_x29',
            'y6_x30',
            'y6_x31',
            'y6_x32',
            'y6_x33',
            'y6_x34',
            'y6_x35',
            'y6_x36',
            'y6_x37',
            'y6_x38',
            'y6_x39',
            'y6_x40',
            'y6_x41',
            'y6_x42',
            'y7_x3',
            'y7_x4',
            'y7_x5',
            'y7_x6',
            'y7_x7',
            'y7_x8',
            'y7_x9',
            'y7_x10',
            'y7_x11',
            'y7_x12',
            'y7_x13',
            'y7_x14',
            'y7_x15',
            'y7_x16',
            'y7_x17',
            'y7_x18',
            'y7_x19',
            'y7_x20',
            'y7_x21',
            'y7_x22',
            'y7_x23',
            'y7_x24',
            'y7_x25',
            'y7_x26',
            'y7_x27',
            'y7_x28',
            'y7_x29',
            'y7_x30',
            'y7_x31',
            'y7_x32',
            'y7_x33',
            'y7_x34',
            'y7_x35',
            'y7_x36',
            'y7_x37',
            'y7_x38',
            'y7_x39',
            'y8_x3',
            'y8_x4',
            'y8_x5',
            'y8_x6',
            'y8_x7',
            'y8_x8',
            'y8_x9',
            'y8_x10',
            'y8_x11',
            'y8_x12',
            'y8_x13',
            'y8_x14',
            'y8_x15',
            'y8_x16',
            'y8_x17',
            'y8_x18',
            'y8_x19',
            'y8_x20',
            'y8_x21',
            'y8_x22',
            'y8_x23',
            'y8_x24',
            'y8_x25',
            'y8_x26',
            'y8_x27',
            'y8_x28',
            'y8_x29',
            'y8_x30',
            'y8_x31',
            'y8_x32',
            'y8_x33',
            'y8_x34',
            'y8_x35',
            'y8_x36',
            'y8_x37',
            'y8_x38',
            'y9_x4',
            'y9_x5',
            'y9_x6',
            'y9_x7',
            'y9_x8',
            'y9_x9',
            'y9_x10',
            'y9_x11',
            'y9_x12',
            'y9_x13',
            'y9_x14',
            'y9_x15',
            'y9_x16',
            'y9_x17',
            'y9_x18',
            'y9_x19',
            'y9_x20',
            'y9_x21',
            'y9_x22',
            'y9_x23',
            'y9_x24',
            'y9_x25',
            'y9_x26',
            'y9_x27',
            'y9_x28',
            'y9_x29',
            'y9_x30',
            'y9_x31',
            'y9_x32',
            'y9_x33',
            'y9_x34',
            'y9_x35',
            'y9_x36',
            'y9_x37',
            'y9_x38',
            'y10_x5',
            'y10_x6',
            'y10_x7',
            'y10_x8',
            'y10_x9',
            'y10_x10',
            'y10_x11',
            'y10_x12',
            'y10_x13',
            'y10_x14',
            'y10_x15',
            'y10_x16',
            'y10_x17',
            'y10_x18',
            'y10_x19',
            'y10_x20',
            'y10_x21',
            'y10_x22',
            'y10_x23',
            'y10_x24',
            'y10_x25',
            'y10_x26',
            'y10_x27',
            'y10_x28',
            'y10_x29',
            'y10_x30',
            'y10_x31',
            'y10_x32',
            'y10_x33',
            'y10_x34',
            'y10_x35',
            'y10_x36',
            'y11_x10',
            'y11_x11',
            'y11_x12',
            'y11_x13',
            'y11_x14',
            'y11_x15',
            'y11_x16',
            'y11_x17',
            'y11_x18',
            'y11_x19',
            'y11_x20',
            'y11_x21',
            'y11_x22',
            'y11_x23',
            'y11_x24',
            'y11_x25',
            'y11_x26',
            'y11_x27',
            'y11_x28',
            'y11_x29',
            'y11_x30',
            'y11_x31',
            'y11_x32',
            'y11_x33',
            'y11_x34',
            'y12_x17',
            'y12_x18',
            'y12_x19',
            'y12_x20',
            'y12_x21',
            'y12_x22',
            'y12_x23',
            'y12_x24',
            'y12_x25',
            'y12_x26',
            'y12_x27',
            'y12_x28',
            'y12_x29',
            'y12_x30',
            'y12_x31',
            'y12_x32',
            'y12_x33',
            'y12_x34',
            'y13_x20',
            'y13_x21',
            'y13_x22',
            'y13_x23',
            'y13_x32',
            'y13_x33',
            'y13_x34',
            'y14_x21',
            'y14_x22',
            'y14_x33',
            'y14_x34',
            'y15_x32',
            'y15_x33',
            'y15_x34',
        ]

    # def generate_valid_region_ids(self) -> list:
    #     # could be refactored - gets all the region_ids and their lat_lon slices
    #     # This was used once to generate the stored list in `valid_region_ids()`
    #     from tqdm import tqdm

    #     region_id_chunk_slices = {}
    #     chunk_info = self.chunk_info
    #     y_starts = chunk_info['y_starts']
    #     x_starts = chunk_info['x_starts']
    #     for iy, _ in enumerate(y_starts):
    #         for ix, _ in enumerate(x_starts):
    #             region_id = f'y{iy}_x{ix}'
    #             y_slice, x_slice = self.region_id_to_latlon_slices(region_id=region_id)
    #             region_id_chunk_slices[region_id] = (y_slice, x_slice)

    #     # For a given region_id, this will check if the data array is empty.
    #     empty_region_ids = []
    #     valid_region_ids = []
    #     for region_id, region_slice in tqdm(region_id_chunk_slices.items()):
    #         subds = self.ds.sel(latitude=region_slice[0], longitude=region_slice[1])
    #         all_null = bool(subds.CRPS.isnull().all().values)
    #         if not all_null:
    #             valid_region_ids.append(region_id)
    #         else:
    #             empty_region_ids.append(region_id)

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

    def chunks_to_slices(self, chunks: dict) -> dict:
        """Create a dict of chunk_ids and slices from input chunk dict

        Parameters
        ----------
        chunks : dict
            Dictionary with chunk sizes for 'longitude' and 'latitude'

        Returns
        -------
        dict
            Dictionary with chunk IDs as keys and corresponding slices as values
        """
        return {key: self.chunk_id_to_slice(value) for key, value in chunks.items()}

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

    def region_id_slice_lookup(self, region_id: str) -> tuple:
        """given a region_id, ex: 'y5_x14, returns the corresponding x,y slices. ex:
        (slice(np.int64(30000), np.int64(36000), None),
        slice(np.int64(85500), np.int64(90000), None))

        Parameters
        ----------
        region_id : str
            The region_id for chunk_id lookup.

        Returns
        -------
        indexer : tuple[slice]
            The corresponding slices (y_slice, x_slice) for the given region_id.
        """
        return self.chunk_id_to_slice(self.region_id_chunk_lookup(region_id))

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

        Parameters
        ----------
        region_id : str
            The region_id for chunk_id lookup.

        Returns
        -------
        latlon_slices : tuple
            (lat_slice, lon_slice)
        """
        chunk_id = self.region_id_chunk_lookup(region_id)
        # Get array slices for this chunk
        y_slice, x_slice = self.chunk_id_to_slice(chunk_id)

        # Convert corners to coordinates
        x_min, y_max = self.index_to_coords(x_slice.start, y_slice.start)  # upper-left
        x_max, y_min = self.index_to_coords(x_slice.stop, y_slice.stop)  # lower-right

        # Create and return the slices
        y_slice = slice(y_max, y_min)
        x_slice = slice(x_min, x_max)

        return (y_slice, x_slice)

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

    def plot_all_chunks(self, color_by_size: bool = False) -> None:
        """
        Plot all data chunks across the entire CONUS with their indices as labels

        Parameters
        ----------
        color_by_size : bool, default False
            If True, color chunks based on their size (useful to identify irregularities)
        """
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # Create figure
        fig, ax = plt.subplots(figsize=(24, 16), subplot_kw={'projection': ccrs.PlateCarree()})

        # Set extent to show CONUS
        print(self.extent_as_tuple)
        ax.set_extent(self.extent_as_tuple, crs=ccrs.PlateCarree())

        # Get chunk information
        chunk_info = self.chunk_info
        y_chunks = chunk_info['y_chunks']
        x_chunks = chunk_info['x_chunks']
        y_starts = chunk_info['y_starts']
        x_starts = chunk_info['x_starts']

        # Track chunk sizes for coloring if needed
        if color_by_size:
            sizes = [
                y_chunks[iy] * x_chunks[ix]
                for iy in range(len(y_chunks))
                for ix in range(len(x_chunks))
            ]
            min_size = min(sizes)
            max_size = max(sizes)

            norm = mcolors.Normalize(vmin=min_size, vmax=max_size)
            cmap = cm.viridis

        # Draw each chunk with label
        for iy, y0 in enumerate(y_starts):
            h = y_chunks[iy]
            for ix, x0 in enumerate(x_starts):
                w = x_chunks[ix]

                # Get chunk boundaries in geographic coordinates
                xx0, yy0 = self.index_to_coords(x0, y0)
                xx1, yy1 = self.index_to_coords(x0 + w, y0 + h)

                # Choose color based on size or use default cycle
                if color_by_size:
                    size = h * w
                    color = cmap(norm(size))

                else:
                    # Use a simple coloring scheme based on indices
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                    color = colors[(iy * len(x_starts) + ix) % len(colors)]

                # Draw rectangle around the chunk
                rect = Rectangle(
                    (xx0, yy1),  # lower left (x, y)
                    xx1 - xx0,  # width
                    yy0 - yy1,  # height
                    transform=ccrs.PlateCarree(),
                    fill=True,
                    facecolor=color,
                    alpha=0.3,
                    edgecolor=color,
                    linewidth=1.5,
                    zorder=10,
                )
                ax.add_patch(rect)
                center_x = (xx0 + xx1) / 2
                center_y = (yy0 + yy1) / 2
                region_id = f'y{iy}_x{ix}'
                ax.text(
                    center_x,
                    center_y,
                    region_id,
                    transform=ccrs.PlateCarree(),
                    ha='center',
                    va='center',
                    fontsize=6,
                    fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                    zorder=20,
                )
        # Add geographic features
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')

        # Add a colorbar if coloring by size
        if color_by_size:
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
            cbar.set_label('Chunk Size (pixels)')

        # Set title
        ax.set_title(
            f'All Chunks ({len(y_chunks)}×{len(x_chunks)} = {len(y_chunks) * len(x_chunks)})'
        )

        plt.tight_layout()
        plt.show()


class VectorConfig(pydantic_settings.BaseSettings):
    """Configuration for vector data processing."""

    environment: Environment = pydantic.Field(
        default=Environment.QA, description='Environment for vector processing'
    )
    version: SemanticVersion | None = pydantic.Field(
        default=None, description='Version of the vector processing pipeline'
    )
    storage_root: str = pydantic.Field(
        ..., description='Root storage path for vector data, can be a bucket name or local path'
    )
    prefix: str | None = pydantic.Field(None, description='Sub-path within the storage root')
    output_prefix: str | None = pydantic.Field(
        None, description='Sub-path within the storage root for pipeline output products'
    )
    debug: bool = pydantic.Field(default=False, description='Enable debugging mode')

    model_config = {'env_prefix': 'ocr_vector_', 'case_sensitive': False}

    def model_post_init(self, __context):
        """Post-initialization to set up prefixes and URIs based on environment."""
        common_part = f'fire-risk/vector/{self.environment.value}'
        if self.prefix is None:
            if self.version:
                self.prefix = f'intermediate/{common_part}/v{self.version}'
            else:
                self.prefix = f'intermediate/{common_part}'

        if self.output_prefix is None:
            if self.version:
                self.output_prefix = f'output/{common_part}/v{self.version}'
            else:
                self.output_prefix = f'output/{common_part}'

        if self.prefix and self.version:
            if f'v{self.version}' not in self.prefix:
                # insert version right before the last part of the prefix
                parts = self.prefix.rsplit('/', 1)
                if len(parts) == 2:
                    self.prefix = f'{parts[0]}/{self.environment.value}/v{self.version}/{parts[1]}'
                else:
                    self.prefix = f'{self.environment.value}/v{self.version}/{self.prefix}'
        if self.output_prefix and self.version:
            if f'v{self.version}' not in self.output_prefix:
                # insert version right before the last part of the prefix
                parts = self.output_prefix.rsplit('/', 1)
                if len(parts) == 2:
                    self.output_prefix = (
                        f'{parts[0]}/{self.environment.value}/v{self.version}/{parts[1]}'
                    )
                else:
                    self.output_prefix = (
                        f'{self.environment.value}/v{self.version}/{self.output_prefix}'
                    )

    def wipe(self):
        """Wipe the vector data storage."""
        if self.debug:
            console.log(
                f'Wiping intermediate vector data storage at {self.storage_root}/{self.prefix}'
            )
        self.delete_region_gpqs()
        self.delete_region_analysis_files()

    # ----------------------------
    # output pmtiles
    # ----------------------------

    @functools.cached_property
    def pmtiles_prefix(self) -> str:
        return f'{self.output_prefix}/pmtiles'

    @functools.cached_property
    def buildings_pmtiles_uri(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.pmtiles_prefix}/buildings.pmtiles')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def tracts_pmtiles_uri(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.pmtiles_prefix}/tracts.pmtiles')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def counties_pmtiles_uri(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.pmtiles_prefix}/counties.pmtiles')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ----------------------------
    # geoparquet
    # ----------------------------

    @functools.cached_property
    def region_geoparquet_prefix(self) -> str:
        return f'{self.prefix}/geoparquet-regions'

    @functools.cached_property
    def geoparquet_prefix(self) -> str:
        return f'{self.output_prefix}/geoparquet'

    @functools.cached_property
    def region_geoparquet_uri(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.region_geoparquet_prefix}')
        path.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def aggregated_region_analysis_prefix(self) -> str:
        return f'{self.output_prefix}/region-analysis'

    @functools.cached_property
    def aggregated_region_analysis_uri(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.aggregated_region_analysis_prefix}')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def building_geoparquet_uri(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.geoparquet_prefix}/buildings.parquet')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def per_region_analysis_prefix(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.output_prefix}/per-region-analysis/')
        path.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def region_summary_stats_prefix(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.output_prefix}/region-summary-stats/')
        path.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def tracts_summary_stats_uri(self) -> UPath:
        """URI for the tracts summary statistics file."""
        geo_table_name = 'tracts'
        return self.region_summary_stats_prefix / f'{geo_table_name}_summary_stats.parquet'

    @functools.cached_property
    def counties_summary_stats_uri(self) -> UPath:
        """URI for the counties summary statistics file."""
        geo_table_name = 'counties'
        return self.region_summary_stats_prefix / f'{geo_table_name}_summary_stats.parquet'

    def upath_delete(self, path: UPath) -> None:
        """Use UPath to handle deletion in a cloud-agnostic way"""
        # First, get a list of all files in the region geoparquet prefix
        if path.exists():
            for file in path.rglob('*'):
                if file.is_file():
                    file.unlink()
        else:
            if self.debug:
                console.log('No files found to delete.')

    def delete_per_region_files(self):
        """Deletes the per region analysis files"""
        if self.debug:
            console.log(
                f'Deleting per region  analysis files from {self.per_region_analysis_prefix}'
            )
        per_region_path = UPath(self.per_region_analysis_prefix)
        self.upath_delete(per_region_path)

    def delete_region_analysis_files(self):
        """Deletes the region aggregated analysis files"""
        if self.debug:
            console.log(
                f'Deleting region aggregated analysis files from {self.aggregated_region_analysis_uri}'
            )
        aggregated_region_path = UPath(self.aggregated_region_analysis_uri)
        self.upath_delete(aggregated_region_path)

    def delete_region_gpqs(self):
        """Delete region geoparquet files from the storage."""
        if self.debug:
            console.log(f'Deleting region geoparquet files from {self.region_geoparquet_uri}')
        if self.region_geoparquet_prefix is None:
            raise ValueError('Region geoparquet prefix must be set before deletion.')
        if 'geoparquet-regions' not in self.region_geoparquet_prefix:
            raise ValueError(
                'It seems like the prefix specified is not the region_id tagged geoparq files. [safety switch]'
            )
        region_path = UPath(self.region_geoparquet_uri)
        self.upath_delete(region_path)

    def pretty_paths(self) -> None:
        """Pretty print key VectorConfig paths and URIs.

        This method intentionally touches cached properties that create
        directories (e.g., via mkdir) so you can verify real locations.
        """
        from rich.panel import Panel
        from rich.table import Table

        def nv(name: str, value: str | None):
            return name, (str(value) if value not in (None, '') else '—')

        rows: list[tuple[str, str]] = []

        # high-level
        rows.append(nv('Environment', getattr(self.environment, 'value', str(self.environment))))
        rows.append(nv('Version', (str(self.version) if self.version else '—')))
        rows.append(nv('Storage root', self.storage_root))

        # prefixes (touch real properties)
        rows.append(nv('Intermediate prefix', self.prefix))
        rows.append(nv('Output prefix', self.output_prefix))
        rows.append(nv('Geoparquet prefix', self.geoparquet_prefix))
        rows.append(nv('Region Geoparquet prefix', self.region_geoparquet_prefix))
        rows.append(nv('PMTiles prefix', self.pmtiles_prefix))

        # derived URIs (touch cached properties that mkdir/prepare parents)
        rows.extend(
            [
                nv('Region Geoparquet URI', str(self.region_geoparquet_uri)),
                nv('Buildings Geoparquet URI', str(self.building_geoparquet_uri)),
                nv('Region summary stats dir', str(self.region_summary_stats_prefix)),
                nv('Tracts summary stats', str(self.tracts_summary_stats_uri)),
                nv('Counties summary stats', str(self.counties_summary_stats_uri)),
                nv('Buildings PMTiles', str(self.buildings_pmtiles_uri)),
                nv('Tracts PMTiles', str(self.tracts_pmtiles_uri)),
                nv('Counties PMTiles', str(self.counties_pmtiles_uri)),
            ]
        )

        table = Table(title=None, show_header=True, header_style='bold magenta')
        table.add_column('Vector setting', style='bold cyan', no_wrap=True)
        table.add_column('Value', style='green')
        for k, v in rows:
            table.add_row(k, v)

        console.print(Panel(table, title='VectorConfig paths', title_align='left'))


class IcechunkConfig(pydantic_settings.BaseSettings):
    """Configuration for icechunk processing."""

    environment: Environment = pydantic.Field(
        default=Environment.QA, description='Environment for icechunk processing'
    )
    version: SemanticVersion | None = pydantic.Field(
        None, description='Version of the icechunk processing pipeline'
    )
    storage_root: str = pydantic.Field(
        ..., description='Root storage path for icechunk data, can be a bucket name or local path'
    )
    prefix: str | None = pydantic.Field(None, description='Sub-path within the storage root')
    debug: bool = pydantic.Field(default=False, description='Enable debugging mode')

    def model_post_init(self, __context):
        """Post-initialization to set up prefixes and URIs based on environment."""
        common_part = f'fire-risk/tensor/{self.environment.value}'
        if self.prefix is None:
            name = 'ocr.icechunk' if self.version is None else f'v{self.version}/ocr.icechunk'
            prefix = f'output/{common_part}/{name}'
            self.prefix = prefix

        if self.prefix and self.version:
            if f'v{self.version}' not in self.prefix:
                # insert version right before the last part of the prefix
                parts = self.prefix.rsplit('/', 1)
                if len(parts) == 2:
                    self.prefix = f'{parts[0]}/{self.environment.value}/v{self.version}/{parts[1]}'
                else:
                    self.prefix = f'{self.environment.value}/v{self.version}/{self.prefix}'

    def wipe(self):
        """Wipe the icechunk repository."""
        if self.debug:
            console.log(f'Wiping icechunk repository at {self.uri}')
        self.delete()
        self.init_repo()

    @functools.cached_property
    def uri(self) -> UPath:
        """Return the URI for the icechunk repository."""
        if self.prefix is None:
            raise ValueError('Prefix must be set before initializing the icechunk repo.')
        return UPath(f'{self.storage_root}/{self.prefix}')

    @functools.cached_property
    def storage(self) -> icechunk.Storage:
        if self.uri is None:
            raise ValueError('URI must be set before initializing the icechunk repo.')

        protocol = self.uri.protocol
        if protocol == 's3':
            parts = self.uri.parts
            bucket = parts[0].strip('/')
            prefix = '/'.join(parts[1:])
            storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
        elif protocol in {'file', 'local'} or protocol == '':
            storage = icechunk.local_filesystem_storage(path=str(self.uri.path))

        else:
            raise ValueError(
                f'Unsupported protocol: {protocol}. Supported protocols are: [s3, file, local]'
            )
        return storage

    def init_repo(self):
        """Creates an icechunk repo or opens if does not exist"""

        icechunk.Repository.open_or_create(self.storage)
        if self.debug:
            console.log('Initialized/Opened icechunk repository')
        commits = self.commit_messages_ancestry()
        if 'initialize store with template' not in commits:
            if self.debug:
                console.log('No template found in icechunk store. Creating a new template dataset.')
            self.create_template()

    def repo_and_session(self, readonly: bool = False, branch: str = 'main') -> dict:
        """Open an icechunk repository and return the session."""
        storage = self.storage
        repo = icechunk.Repository.open(storage)
        if readonly:
            session = repo.readonly_session(branch=branch)
        else:
            session = repo.writable_session(branch=branch)

        if self.debug:
            console.log(
                f'Opened icechunk repository at {self.uri} with branch {branch} in {"readonly" if readonly else "writable"} mode.'
            )
        return {'repo': repo, 'session': session}

    def delete(self):
        """Delete the icechunk repository."""
        if self.uri is None:
            raise ValueError('URI must be set before deleting the icechunk repo.')

        console.log(f'Deleting icechunk repository at {self.uri}')
        if self.uri.protocol == 's3':
            if self.uri.exists():
                self.uri.rmdir()
            else:
                if self.debug:
                    console.log('No files found to delete.')

        elif self.uri.protocol in {'file', 'local'} or self.uri.protocol == '':
            path = self.uri.path
            import shutil

            if UPath(path).exists():
                shutil.rmtree(path)
            else:
                if self.debug:
                    console.log('No files found to delete.')

        if self.debug:
            console.log('Deleted icechunk repository')

    def create_template(self):
        """Create a template dataset for icechunk store"""
        import dask
        import dask.array
        import numpy as np
        import xarray as xr

        repo_and_session = self.repo_and_session()
        # NOTE: This is hardcoded as using the USFS 30m chunking scheme!
        config = ChunkingConfig()

        ds = config.ds
        ds['CRPS'].encoding = {}

        template = xr.Dataset(ds.coords).drop_vars('spatial_ref')
        var_encoding_dict = {
            'chunks': ((config.chunks['latitude'], config.chunks['longitude'])),
            'fill_value': np.nan,
        }
        template_data_array = xr.DataArray(
            dask.array.empty(
                (config.ds.sizes['latitude'], config.ds.sizes['longitude']),
                dtype='float32',
                chunks=-1,
            ),
            dims=('latitude', 'longitude'),
        )

        variables = [
            'wind_risk_2011',
            'wind_risk_2047',
            'burn_probability_2011',
            'burn_probability_2047',
            'conditional_risk_usfs',
            'burn_probability_usfs_2011',
            'burn_probability_usfs_2047',
        ]
        template_encoding_dict = {}
        for variable in variables:
            template[variable] = template_data_array
            template_encoding_dict[variable] = var_encoding_dict
        template.to_zarr(
            repo_and_session['session'].store,
            compute=False,
            mode='w',
            encoding=template_encoding_dict,
            consolidated=False,
        )
        repo_and_session['session'].commit('initialize store with template')
        if self.debug:
            console.log('Created icechunk template')

    def commit_messages_ancestry(self, branch: str = 'main') -> list[str]:
        """Get the commit messages ancestry for the icechunk repository."""
        repo_and_session = self.repo_and_session(readonly=True)
        repo = repo_and_session['repo']

        commit_messages = [commit.message for commit in list(repo.ancestry(branch=branch))]
        # separate commits by ',' and handle case of single length ancestry commit history

        split_commits = [
            msg
            for message in commit_messages
            for msg in (message.split(',') if ',' in message else [message])
        ]
        return split_commits

    def region_id_exists(self, region_id: str, *, branch: str = 'main') -> bool:
        region_ids_in_ancestry = self.commit_messages_ancestry(branch=branch)

        if region_id in region_ids_in_ancestry:
            return True

        return False

    def processed_regions(self, *, branch: str = 'main') -> list[str]:
        """Get a list of region IDs that have already been processed."""
        region_ids = set()
        for message in self.commit_messages_ancestry(branch=branch):
            if message.startswith('wrote region_id'):
                region_ids.add(message.split('(')[1].split(')')[0])
        result = sorted(region_ids)
        if self.debug:
            console.log(f'Found processed {len(result)} region IDs: {result}')
        return result

    def insert_region_uncooperative(
        self, subset_ds: xr.Dataset, *, region_id: str, branch: str = 'main'
    ):
        """Insert region into Icechunk store

        Parameters
        ----------
        subset_ds : xr.Dataset
            The subset dataset to insert into the Icechunk store.
        region_id : str
            The region ID corresponding to the subset dataset.
        branch : str, optional
            The branch to use in the Icechunk repository, by default 'main'.

        """

        if self.debug:
            console.log(f'Inserting region: {region_id} into Icechunk store: ')

        while True:
            try:
                session = self.repo_and_session(readonly=False, branch=branch)['session']
                subset_ds.to_zarr(
                    session.store,
                    region='auto',
                    consolidated=False,
                )
                # Trying out the rebase strategy described here: https://github.com/earth-mover/icechunk/discussions/802#discussioncomment-13064039
                # We should be in the same position, where we don't have real conflicts, just write timing conflicts.
                session.commit(
                    f'wrote region_id ({region_id})', rebase_with=icechunk.ConflictDetector()
                )
                if self.debug:
                    console.log(f'Wrote dataset: {subset_ds} to region: {region_id}')
                break

            except Exception as exc:
                delay = random.uniform(3.0, 10.0)
                if self.debug:
                    console.log(f'Conflict detected while writing region {region_id}: {exc}')
                    console.log(f'retrying to write region_id: {region_id} in {delay:.2f}s')

                time.sleep(delay)
                pass

    def pretty_paths(self) -> None:
        """Pretty print key IcechunkConfig paths and URIs.

        This version touches cached properties (e.g., uri, storage) to
        surface real configuration and types.
        """
        from rich.panel import Panel
        from rich.table import Table

        def nv(name: str, value: str | None):
            return name, (str(value) if value not in (None, '') else '—')

        rows: list[tuple[str, str]] = []
        rows.append(nv('Environment', getattr(self.environment, 'value', str(self.environment))))
        rows.append(nv('Version', (str(self.version) if self.version else '—')))
        rows.append(nv('Storage root', self.storage_root))
        rows.append(nv('Prefix', self.prefix))

        # Touch real cached properties
        uri = self.uri
        rows.append(nv('Repository URI', str(uri)))
        rows.append(nv('Protocol', uri.protocol or 'file'))

        table = Table(title=None, show_header=True, header_style='bold magenta')
        table.add_column('Icechunk setting', style='bold cyan', no_wrap=True)
        table.add_column('Value', style='green')
        for k, v in rows:
            table.add_row(k, v)

        console.print(Panel(table, title='IcechunkConfig paths', title_align='left'))


@dataclass
class RegionIDStatus:
    provided_region_ids: set[str]
    valid_region_ids: set[str]
    invalid_region_ids: set[str]
    processed_region_ids: set[str]
    previously_processed_ids: set[str]
    unprocessed_valid_region_ids: set[str]


class OCRConfig(pydantic_settings.BaseSettings):
    """Configuration settings for OCR processing."""

    environment: Environment = pydantic.Field(
        default=Environment.QA, description='Environment for OCR processing'
    )
    version: SemanticVersion | None = pydantic.Field(
        default=None,
        description=(
            'Optional semantic version (e.g., 1.2.3 or v1.2.3). When provided, appended to '
            'intermediate and output prefixes for versioned storage.'
        ),
    )
    storage_root: str = pydantic.Field(
        ..., description='Root storage path for OCR data, can be a bucket name or local path'
    )

    vector: VectorConfig | None = pydantic.Field(None, description='Vector configuration')
    icechunk: IcechunkConfig | None = pydantic.Field(None, description='Icechunk configuration')
    chunking: ChunkingConfig | None = pydantic.Field(
        None, description='Chunking configuration for OCR processing'
    )
    coiled: CoiledConfig | None = pydantic.Field(None, description='Coiled configuration')
    debug: bool = pydantic.Field(False, description='Enable debugging mode')

    model_config = {'env_prefix': 'ocr_', 'case_sensitive': False}

    def model_post_init(self, __context):
        # Pass environment and wipe to VectorConfig if not already set
        if self.vector is None:
            object.__setattr__(
                self,
                'vector',
                VectorConfig(
                    storage_root=self.storage_root,
                    environment=self.environment,
                    debug=self.debug,
                    version=self.version,
                ),
            )
        if self.icechunk is None:
            object.__setattr__(
                self,
                'icechunk',
                IcechunkConfig(
                    storage_root=self.storage_root,
                    environment=self.environment,
                    debug=self.debug,
                    version=self.version,
                ),
            )
        if self.chunking is None:
            object.__setattr__(
                self,
                'chunking',
                ChunkingConfig(debug=self.debug),
            )

        if self.coiled is None:
            object.__setattr__(
                self,
                'coiled',
                CoiledConfig(),
            )

    def pretty_paths(self) -> None:
        """Pretty print key OCRConfig paths and URIs.

        This method intentionally touches cached properties that create
        directories (e.g., via mkdir) so you can verify real locations.
        """
        from rich.panel import Panel
        from rich.table import Table

        def nv(name: str, value: str | None):
            return name, (str(value) if value not in (None, '') else '—')

        rows: list[tuple[str, str]] = []

        # high-level
        rows.append(nv('Environment', getattr(self.environment, 'value', str(self.environment))))
        rows.append(nv('Version', (str(self.version) if self.version else '—')))
        rows.append(nv('Storage root', self.storage_root))

        table = Table(title=None, show_header=True, header_style='bold magenta')
        table.add_column('OCR setting', style='bold cyan', no_wrap=True)
        table.add_column('Value', style='green')
        for k, v in rows:
            table.add_row(k, v)

        console.print(Panel(table, title='OCRConfig paths', title_align='left'))

        if self.vector:
            self.vector.pretty_paths()
        if self.icechunk:
            self.icechunk.pretty_paths()

    # ------------------------------------------------------------------
    # Region ID selection / validation helpers (used by CLI pipeline)
    # ------------------------------------------------------------------
    def _compose_region_id_error(self, status: 'RegionIDStatus') -> str:
        """Compose a detailed error message mirroring previous CLI behavior.

        Parameters
        ----------
        status : RegionIDStatus
            Computed status object.
        """
        error_message = 'No valid region IDs to process. All provided region IDs were rejected for the following reasons:\n'
        # Ensure required sub-config present (defensive; model_post_init guarantees this)
        assert self.chunking is not None, 'Chunking configuration not initialized'
        if status.invalid_region_ids:
            error_message += (
                f'- Invalid region IDs: {", ".join(sorted(status.invalid_region_ids))}\n'
            )
            # include (truncated) list of valid ids for reference
            error_message += (
                '  Valid region IDs: '
                f'{", ".join(sorted(list(self.chunking.valid_region_ids)))}...\n'
            )
        if status.previously_processed_ids:
            error_message += (
                '- Already processed region IDs: '
                f'{", ".join(sorted(status.previously_processed_ids))}\n'
            )
        error_message += "\nPlease provide valid region IDs that haven't been processed yet."
        return error_message

    def resolve_region_ids(self, provided_region_ids: set[str]) -> 'RegionIDStatus':
        """Validate provided region IDs against valid + processed sets.

        Returns a RegionIDStatus object or raises ValueError if none are processable.
        """
        assert self.chunking is not None, 'Chunking configuration not initialized'
        assert self.icechunk is not None, 'Icechunk configuration not initialized'
        all_valid = set(self.chunking.valid_region_ids)
        valid_region_ids = provided_region_ids.intersection(all_valid)
        processed_region_ids = set(self.icechunk.processed_regions())
        unprocessed_valid_region_ids = valid_region_ids.difference(processed_region_ids)
        invalid_region_ids = provided_region_ids.difference(all_valid)
        previously_processed_ids = provided_region_ids.intersection(processed_region_ids)
        status = RegionIDStatus(
            provided_region_ids=provided_region_ids,
            valid_region_ids=valid_region_ids,
            invalid_region_ids=invalid_region_ids,
            processed_region_ids=processed_region_ids,
            previously_processed_ids=previously_processed_ids,
            unprocessed_valid_region_ids=unprocessed_valid_region_ids,
        )
        if len(unprocessed_valid_region_ids) == 0:
            raise ValueError(self._compose_region_id_error(status))
        return status

    def select_region_ids(
        self, region_ids: list[str] | None, *, all_region_ids: bool = False
    ) -> 'RegionIDStatus':
        """Helper to pick the effective set of region IDs (all or user-provided) and
        return the validated status object.
        """
        assert self.chunking is not None, 'Chunking configuration not initialized'
        provided = set(self.chunking.valid_region_ids) if all_region_ids else set(region_ids or [])
        return self.resolve_region_ids(provided)


def load_config(file_path: Path | None) -> OCRConfig:
    """Load OCR configuration from an env file (dotenv) or current environment."""
    if file_path is None:
        return OCRConfig()
    dotenv.load_dotenv(file_path)
    return OCRConfig()
