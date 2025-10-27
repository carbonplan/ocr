import functools
import random
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dotenv
import icechunk
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import odc.geo.xr  # noqa
import pydantic
import pydantic_settings
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pydantic_extra_types.semantic_version import SemanticVersion
from shapely.geometry import Polygon, box
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
    def extent_as_tuple_5070(self):
        """Get extent in EPSG:5070 projection as tuple (xmin, xmax, ymin, ymax)"""
        from pyproj import Transformer

        bounds = self.extent.bounds
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)

        # Transform corner points
        xmin_5070, ymin_5070 = transformer.transform(bounds[0], bounds[1])
        xmax_5070, ymax_5070 = transformer.transform(bounds[2], bounds[3])

        return (xmin_5070, xmax_5070, ymin_5070, ymax_5070)

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
        """Generate valid region IDs by checking which regions contain non-null data.

        Returns
        -------
        list
            List of valid region IDs (e.g., 'y1_x3', 'y2_x4', etc.)
        """
        import json

        # Use cache file in the package directory
        cache_file = Path(__file__).parent / 'data' / 'valid_region_ids.json'

        # Try to load from cache first
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached_data = json.load(f)
                if self.debug:
                    console.log(
                        f'Loaded {len(cached_data)} valid region IDs from cache: {cache_file}'
                    )
                return cached_data
            except Exception as e:
                if self.debug:
                    console.log(f'Failed to load cache file: {e}. Regenerating...')

        # If cache doesn't exist or failed to load, compute valid region IDs
        chunk_info = self.chunk_info
        y_starts = chunk_info['y_starts']
        x_starts = chunk_info['x_starts']

        if self.debug:
            console.log('Computing valid region IDs (this may take a while)...')

        valid_region_ids = []
        for iy, _ in enumerate(y_starts):
            for ix, _ in enumerate(x_starts):
                region_id = f'y{iy}_x{ix}'
                y_slice, x_slice = self.region_id_to_latlon_slices(region_id=region_id)
                subds = self.ds.sel(latitude=y_slice, longitude=x_slice)
                all_null = bool(subds.CRPS.isnull().all().values)
                if not all_null:
                    valid_region_ids.append(region_id)

        # Save to cache for future use
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(valid_region_ids, f, indent=2)

        if self.debug:
            console.log(
                f'Computed and cached {len(valid_region_ids)} valid region IDs to {cache_file}'
            )

        return valid_region_ids

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

        # Create figure
        fig, ax = plt.subplots(figsize=(24, 16), subplot_kw={'projection': ccrs.PlateCarree()})

        # Set extent to show CONUS
        ax.set_extent(self.extent_as_tuple, crs=ccrs.PlateCarree())

        # Get chunk information
        chunk_info = self.chunk_info
        y_chunks = chunk_info['y_chunks']
        x_chunks = chunk_info['x_chunks']
        y_starts = chunk_info['y_starts']
        x_starts = chunk_info['x_starts']

        # Track chunk sizes for coloring if needed
        norm = None
        cmap = None
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
                if color_by_size and cmap is not None and norm is not None:
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

    def bbox_from_wgs84(self, xmin: float, ymin: float, xmax: float, ymax: float):
        "https://observablehq.com/@rdmurphy/u-s-state-bounding-boxes"

        # Create and return bounding box in EPSG:4326 (WGS84)
        # This matches the coordinate system of the data
        bbox = box(xmin, ymin, xmax, ymax)
        return bbox

    def get_chunks_for_bbox(self, bbox: Polygon | tuple) -> list[tuple[int, int]]:
        """
        Find all chunks that intersect with the given bounding box

        Parameters
        ----------
        bbox : BoundingBox or tuple
            Bounding box to check for intersection. If tuple, format is (minx, miny, maxx, maxy)

        Returns
        -------
        list of tuples
            List of (iy, ix) tuples identifying the intersecting chunks
        """
        # Convert tuple to BoundingBox if needed
        if isinstance(bbox, tuple):
            if len(bbox) == 4:
                bbox = box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
            else:
                raise ValueError('Bounding box tuple must have 4 elements (minx, miny, maxx, maxy)')

        # Get chunk info
        chunk_info = self.chunk_info
        y_chunks = chunk_info['y_chunks']
        x_chunks = chunk_info['x_chunks']
        y_starts = chunk_info['y_starts']
        x_starts = chunk_info['x_starts']

        # Find intersecting chunks
        intersecting_chunks = []

        for iy, y0 in enumerate(y_starts):
            h = y_chunks[iy]
            for ix, x0 in enumerate(x_starts):
                w = x_chunks[ix]

                # Get chunk boundaries in geographic coordinates
                xx0, yy0 = self.index_to_coords(x0, y0)
                xx1, yy1 = self.index_to_coords(x0 + w, y0 + h)

                # Create a box for this chunk (note Y axis flip)
                chunk_box = box(xx0, yy1, xx1, yy0)

                # Check for intersection
                if bbox.intersects(chunk_box):
                    intersecting_chunks.append((iy, ix))

        return intersecting_chunks

    def visualize_chunks_on_conus(
        self,
        chunks: list[tuple[int, int]] | None = None,
        color_by_size: bool = False,
        highlight_chunks: list[tuple[int, int]] | None = None,
        include_all_chunks: bool = False,
    ) -> None:
        """
        Visualize specified chunks on CONUS map

        Parameters
        ----------
        chunks : list of tuples, optional
            List of (iy, ix) tuples specifying chunks to visualize
            If None, will show all chunks
        color_by_size : bool, default False
            If True, color chunks based on their size
        highlight_chunks : list of tuples, optional
            List of (iy, ix) tuples specifying chunks to highlight
        include_all_chunks : bool, default False
            If True, show all chunks in background with low opacity
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})

        # Set extent - either full CONUS or custom extent
        ax.set_extent(self.extent_as_tuple, crs=ccrs.PlateCarree())

        # Get chunk information
        chunk_info = self.chunk_info
        y_chunks = chunk_info['y_chunks']
        x_chunks = chunk_info['x_chunks']
        y_starts = chunk_info['y_starts']
        x_starts = chunk_info['x_starts']

        # Set up colors
        norm = None
        cmap = None
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

        # Default to all chunks if none specified
        if chunks is None:
            chunks = [(iy, ix) for iy in range(len(y_chunks)) for ix in range(len(x_chunks))]

        # Draw background chunks if requested
        if include_all_chunks and chunks != [
            (iy, ix) for iy in range(len(y_chunks)) for ix in range(len(x_chunks))
        ]:
            for iy, y0 in enumerate(y_starts):
                h = y_chunks[iy]
                for ix, x0 in enumerate(x_starts):
                    # Skip chunks that are in the main visualization
                    if (iy, ix) in chunks:
                        continue

                    w = x_chunks[ix]
                    xx0, yy0 = self.index_to_coords(x0, y0)
                    xx1, yy1 = self.index_to_coords(x0 + w, y0 + h)

                    rect = Rectangle(
                        (xx0, yy1),
                        xx1 - xx0,
                        yy0 - yy1,
                        transform=ccrs.PlateCarree(),
                        fill=True,
                        facecolor='lightgray',
                        alpha=0.2,
                        edgecolor='gray',
                        linewidth=0.5,
                        zorder=5,
                    )
                    ax.add_patch(rect)

        # Draw the specified chunks with proper styling
        for iy, ix in chunks:
            y0 = y_starts[iy]
            h = y_chunks[iy]
            x0 = x_starts[ix]
            w = x_chunks[ix]

            # Get chunk boundaries in geographic coordinates
            xx0, yy0 = self.index_to_coords(x0, y0)
            xx1, yy1 = self.index_to_coords(x0 + w, y0 + h)

            # Determine styling
            is_highlighted = highlight_chunks is not None and (iy, ix) in highlight_chunks

            # Choose color based on size or use default cycle
            if is_highlighted:
                color = 'red'
                fill_alpha = 0.4
                linewidth = 2.0
                zorder = 15
            elif color_by_size and cmap is not None and norm is not None:
                size = h * w
                color = cmap(norm(size))
                # edge_alpha = 0.8
                fill_alpha = 0.3
                linewidth = 1.5
                zorder = 10
            else:
                # Use a simple coloring scheme based on indices
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                color = colors[(iy * len(x_starts) + ix) % len(colors)]
                fill_alpha = 0.3
                linewidth = 1.5
                zorder = 10

            # Draw rectangle around the chunk
            rect = Rectangle(
                (xx0, yy1),  # lower left (x, y)
                xx1 - xx0,  # width
                yy0 - yy1,  # height
                transform=ccrs.PlateCarree(),
                fill=True,
                facecolor=color,
                alpha=fill_alpha,
                edgecolor=color,
                linewidth=linewidth,
                zorder=zorder,
            )
            ax.add_patch(rect)

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
        if len(chunks) == len(y_chunks) * len(x_chunks):
            ax.set_title(f'All Chunks ({len(y_chunks)}×{len(x_chunks)} = {len(chunks)})')
        else:
            ax.set_title(
                f'Selected Chunks ({len(chunks)} of {len(y_chunks)}×{len(x_chunks)} total)'
            )

        # Add a legend
        legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Selected Chunks')]
        if highlight_chunks:
            legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Highlighted Chunks'))
        if include_all_chunks:
            legend_elements.append(Line2D([0], [0], color='gray', lw=1, label='Other Chunks'))

        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.show()


class PyramidConfig(pydantic_settings.BaseSettings):
    """Configuration for visualization pyramid / multiscales"""

    environment: Environment = pydantic.Field(
        default=Environment.QA, description='Environment for pyramid'
    )
    version: SemanticVersion | None = pydantic.Field(
        default=None, description='Version of the pyramid processing pipeline'
    )
    storage_root: str = pydantic.Field(
        ..., description='Root storage path for pyramid. can be a bucket name or local path'
    )
    output_prefix: str | None = pydantic.Field(
        None, description='Sub-path within the storage root for pipeline output products'
    )
    debug: bool = pydantic.Field(default=False, description='Enable debugging mode')
    model_config = {'env_prefix': 'ocr_vector_', 'case_sensitive': False}

    def model_post_init(self, __context):
        """Post-initialization to set up prefixes and URIs based on environment."""
        common_part = f'fire-risk/pyramid/{self.environment.value}'

        if self.output_prefix is None:
            if self.version:
                self.output_prefix = f'output/{common_part}/v{self.version}/pyramid.zarr'
            else:
                self.output_prefix = f'output/{common_part}/pyramid.zarr'

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

    @property
    def pyramid_uri(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.output_prefix}')
        return path

    def wipe(self):
        """Wipe the pyramid data storage."""
        if self.debug:
            console.log(f'Wiping pyramid data:\n- {self.pyramid_uri.parent}\n')

        self.upath_delete(self.pyramid_uri.parent)


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
                f'Wiping vector data storage at these locations:\n'
                f'- {self.building_geoparquet_uri.parent}\n'
                f'- {self.buildings_pmtiles_uri.parent}\n'
                f'- {self.region_geoparquet_uri}\n'
                f'- {self.aggregated_region_analysis_uri}\n'
                f'- {self.tracts_summary_stats_uri.parent}\n'
            )
        self.upath_delete(self.building_geoparquet_uri.parent)
        self.upath_delete(self.buildings_pmtiles_uri.parent)
        self.upath_delete(self.region_geoparquet_uri)
        self.upath_delete(self.aggregated_region_analysis_uri)
        self.upath_delete(self.tracts_summary_stats_uri.parent)

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
    def region_pmtiles_uri(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.pmtiles_prefix}/regions.pmtiles')
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
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def building_geoparquet_uri(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.geoparquet_prefix}/buildings.parquet')
        path.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def region_summary_stats_prefix(self) -> UPath:
        path = UPath(f'{self.storage_root}/{self.output_prefix}/region-summary-stats/')
        path.mkdir(parents=True, exist_ok=True)
        return path

    @functools.cached_property
    def block_summary_stats_uri(self) -> UPath:
        """URI for the block summary statistics file."""
        geo_table_name = 'blocks'
        return self.region_summary_stats_prefix / f'{geo_table_name}_summary_stats.parquet'

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
        if not path.exists():
            if self.debug:
                console.log('No files found to delete.')
            return

        protocol = path.protocol

        # For S3, use fsspec's rm method which supports recursive deletion
        if protocol == 's3':
            if self.debug:
                console.log(f'Deleting S3 path: {path}')
            # Use the underlying filesystem's rm method for efficient batch deletion
            fs = path.fs
            fs.rm(path.path, recursive=True)
        else:
            # For local filesystems, use standard recursive deletion
            if self.debug:
                console.log(f'Deleting local path: {path}')
            import shutil

            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

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
                nv('Buildings Geoparquet', str(self.building_geoparquet_uri)),
                nv('Region summary stats dir', str(self.region_summary_stats_prefix)),
                nv('Block summary stats', str(self.block_summary_stats_uri)),
                nv('Tracts summary stats', str(self.tracts_summary_stats_uri)),
                nv('Counties summary stats', str(self.counties_summary_stats_uri)),
                nv('Buildings PMTiles', str(self.buildings_pmtiles_uri)),
                nv('Region PMTiles', str(self.region_pmtiles_uri)),
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

        if self.uri.protocol == 's3':
            if self.uri.exists():
                # Use the underlying filesystem's rm method for efficient batch deletion
                fs = self.uri.fs
                console.log(f'Deleting icechunk repository at {self.uri}')
                fs.rm(self.uri.path, recursive=True)
            else:
                if self.debug:
                    console.log('No files found to delete.')

        elif self.uri.protocol in {'file', 'local'} or self.uri.protocol == '':
            path = self.uri.path
            import shutil

            if UPath(path).exists():
                console.log(f'Deleting icechunk repository at {self.uri}')
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
            'USFS_RPS',
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
    pyramid: PyramidConfig | None = pydantic.Field(None, description='Pyramid configuration')
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
        if self.pyramid is None:
            object.__setattr__(
                self,
                'pyramid',
                PyramidConfig(
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
