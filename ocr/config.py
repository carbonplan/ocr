import functools

import icechunk
import numpy as np
import pydantic
import pydantic_settings
from upath import UPath

from ocr import catalog
from ocr.console import console
from ocr.icechunk_utils import get_commit_messages_ancestry
from ocr.types import Branch


class ChunkingConfig(pydantic_settings.BaseSettings):
    chunks: dict | None = pydantic.Field(None, description='Chunk sizes for longitude and latitude')

    class Config:
        """Configuration for Pydantic settings."""

        env_prefix = 'ocr_chunking_'
        case_sensitive = False

    def model_post_init(self, __context):
        self.chunks = self.chunks or dict(
            zip(['longitude', 'latitude'], self.ds['CRPS'].data.chunksize)
        )

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
        import odc.geo.xr  # noqa

        dataset = (
            catalog.get_dataset('USFS-wildfire-risk-communities-4326')
            .to_xarray()
            .astype('float32')[['CRPS']]
        )
        dataset = dataset.odc.assign_crs('epsg:4326')
        return dataset

    @functools.cached_property
    def transform(self):
        import odc.geo.xr  # noqa

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

    def generate_valid_region_ids(self) -> list:
        # could be refactored - gets all the region_ids and their lat_lon slices
        # This was used once to generate the stored list in `valid_region_ids()`
        from tqdm import tqdm

        region_id_chunk_slices = {}
        chunk_info = self.chunk_info
        y_starts = chunk_info['y_starts']
        x_starts = chunk_info['x_starts']
        for iy, _ in enumerate(y_starts):
            for ix, _ in enumerate(x_starts):
                region_id = f'y{iy}_x{ix}'
                y_slice, x_slice = self.region_id_to_latlon_slices(region_id=region_id)
                region_id_chunk_slices[region_id] = (y_slice, x_slice)

        # For a given region_id, this will check if the data array is empty.
        empty_region_ids = []
        valid_region_ids = []
        for region_id, region_slice in tqdm(region_id_chunk_slices.items()):
            subds = self.ds.sel(latitude=region_slice[0], longitude=region_slice[1])
            all_null = bool(subds.CRPS.isnull().all().values)
            if not all_null:
                valid_region_ids.append(region_id)
            else:
                empty_region_ids.append(region_id)

    def index_to_coords(self, x_idx, y_idx):
        """Convert array indices to EPSG:4326 coordinates"""
        x, y = self.transform * (x_idx, y_idx)
        return x, y

    def chunks_to_slices(self, chunks: dict) -> dict:
        """Create a dict of chunk_ids and slices from input chunk dict"""
        return {key: self.chunk_id_to_slice(value) for key, value in chunks.items()}

    def region_id_chunk_lookup(self, region_id: str) -> tuple:
        """given a region_id, ex: 'y5_x14, returns the corresponding chunk (5, 14)"""
        return self.get_chunk_mapping()[region_id]

    def region_id_slice_lookup(self, region_id: str) -> tuple:
        """given a region_id, ex: 'y5_x14, returns the corresponding x,y slices. ex:
        (slice(np.int64(30000), np.int64(36000), None),
        slice(np.int64(85500), np.int64(90000), None))"""
        return self.chunk_id_to_slice(self.region_id_chunk_lookup(region_id))

    def chunk_id_to_slice(self, chunk_id):
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
        tuple[slice]
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

    def region_id_to_latlon_slices(self, region_id):
        """
        Get latitude and longitude slices from region_id

        Parameters
        ----------
        region_id : tuple
            The region_id for chunk_id lookup.

        Returns
        -------
        tuple
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

    def get_chunk_mapping(self):
        """Returns a dict of region_ids and their corresponding chunk_indexes."""
        chunk_info = self.chunk_info
        y_starts = chunk_info['y_starts']
        x_starts = chunk_info['x_starts']

        chunk_mapping = {}
        for iy, y0 in enumerate(y_starts):
            for ix, x0 in enumerate(x_starts):
                chunk_mapping[f'y{iy}_x{ix}'] = (iy, ix)

        return chunk_mapping

    def plot_all_chunks(self, color_by_size=False):
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
                region_id = f'y{iy}_x{ix}'  # Match your get_chunk_mapping format
                ax.text(
                    center_x,
                    center_y,
                    region_id,
                    transform=ccrs.PlateCarree(),
                    ha='center',
                    va='center',
                    fontsize=8,
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

    branch: Branch = pydantic.Field(default=Branch.QA, description='Branch for vector processing')
    storage_root: str = pydantic.Field(
        ..., description='Root storage path for vector data, can be a bucket name or local path'
    )
    prefix: str | None = pydantic.Field(None, description='Sub-path within the storage root')

    class Config:
        """Configuration for Pydantic settings."""

        env_prefix = 'ocr_vector_'
        case_sensitive = False

    def model_post_init(self, __context):
        """Post-initialization to set up prefixes and URIs based on branch."""
        if self.prefix is None:
            self.prefix = f'intermediate/fire-risk/vector/{self.branch.value}'

    def wipe(self):
        """Wipe the vector data storage."""
        console.log(f'Wiping vector data storage at {self.storage_root}/{self.prefix}')
        self.delete_region_gpqs()

    @functools.cached_property
    def region_geoparquet_prefix(self) -> str:
        return f'{self.prefix}/geoparquet-regions/'

    @functools.cached_property
    def region_geoparquet_uri(self) -> UPath:
        return UPath(f'{self.storage_root}/{self.region_geoparquet_prefix}')

    @functools.cached_property
    def consolidated_geoparquet_prefix(self) -> str:
        return f'{self.prefix}/consolidated-geoparquet.parquet'

    @functools.cached_property
    def consolidated_geoparquet_uri(self) -> UPath:
        return UPath(f'{self.storage_root}/{self.consolidated_geoparquet_prefix}')

    @functools.cached_property
    def pmtiles_prefix(self) -> str:
        return f'{self.prefix}/consolidated.pmtiles'

    @functools.cached_property
    def pmtiles_prefix_uri(self) -> UPath:
        return UPath(f'{self.storage_root}/{self.pmtiles_prefix}')

    @functools.cached_property
    def aggregated_regions_prefix(self) -> UPath:
        return UPath(f'{self.storage_root}/{self.prefix}/aggregated-regions/')

    @functools.cached_property
    def tracts_summary_stats_uri(self) -> UPath:
        """URI for the tracts summary statistics file."""
        geo_table_name = 'tracts'
        return self.aggregated_regions_prefix / f'{geo_table_name}_summary_stats.parquet'

    @functools.cached_property
    def counties_summary_stats_uri(self) -> UPath:
        """URI for the counties summary statistics file."""
        geo_table_name = 'counties'
        return self.aggregated_regions_prefix / f'{geo_table_name}_summary_stats.parquet'

    def delete_region_gpqs(self):
        """Delete region geoparquet files from the storage."""
        console.log(f'Deleting region geoparquet files from {self.region_geoparquet_uri}')
        if self.region_geoparquet_prefix is None:
            raise ValueError('Region geoparquet prefix must be set before deletion.')
        if 'geoparquet-regions' not in self.region_geoparquet_prefix:
            raise ValueError(
                'It seems like the prefix specified is not the region_id tagged geoparq files. [safety switch]'
            )

        # Use UPath to handle deletion in a cloud-agnostic way
        # First, get a list of all files in the region geoparquet prefix
        region_path = UPath(self.region_geoparquet_uri)
        if region_path.exists():
            for file in region_path.glob('*'):
                if file.is_file():
                    file.unlink()
        else:
            console.log('No files found to delete.')


class IcechunkConfig(pydantic_settings.BaseSettings):
    """Configuration for icechunk processing."""

    branch: Branch = pydantic.Field(default=Branch.QA, description='Branch for icechunk processing')
    storage_root: str = pydantic.Field(
        ..., description='Root storage path for icechunk data, can be a bucket name or local path'
    )
    prefix: str | None = pydantic.Field(None, description='Sub-path within the storage root')

    def model_post_init(self, __context):
        """Post-initialization to set up prefixes and URIs based on branch."""
        if self.prefix is None:
            self.prefix = f'intermediate/fire-risk/tensor/{self.branch.value}/template.icechunk'
        self.init_repo()

    def wipe(self):
        """Wipe the icechunk repository."""
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
        console.log('Initialized/Opened icechunk repository')
        commits = get_commit_messages_ancestry(self.repo_and_session()['repo'])
        if 'initialize store with template' not in commits:
            console.log('No template found in icechunk store. Creating a new template dataset.')
            self.create_template()

    def repo_and_session(self, readonly: bool = False, branch: str = 'main'):
        """Open an icechunk repository and return the session."""
        storage = self.storage
        repo = icechunk.Repository.open_or_create(storage)
        if readonly:
            session = repo.readonly_session(branch=branch)
        else:
            session = repo.writable_session(branch=branch)

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
                for file in self.uri.glob('*'):
                    if file.is_file():
                        file.unlink()
                self.uri.rmdir()
            else:
                console.log('No files found to delete.')

        elif self.uri.protocol in {'file', 'local'} or self.uri.protocol == '':
            path = self.uri.path
            import shutil

            if UPath(path).exists():
                shutil.rmtree(path)
            else:
                console.log('No files found to delete.')

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
            'chunks': ((config.chunks['longitude'], config.chunks['latitude'])),
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
        variables = ['risk_2011', 'risk_2047', 'wind_risk_2011', 'wind_risk_2047']
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
        console.log('Created icechunk template')


class OCRConfig(pydantic_settings.BaseSettings):
    """Configuration settings for OCR processing."""

    branch: Branch = pydantic.Field(default=Branch.QA, description='Branch for OCR processing')
    storage_root: str = pydantic.Field(
        ..., description='Root storage path for OCR data, can be a bucket name or local path'
    )
    vector: VectorConfig | None = pydantic.Field(None, description='Vector configuration')
    icechunk: IcechunkConfig | None = pydantic.Field(None, description='Icechunk configuration')
    chunking: ChunkingConfig | None = pydantic.Field(
        None, description='Chunking configuration for OCR processing'
    )

    class Config:
        """Configuration for Pydantic settings."""

        env_prefix = 'ocr_'
        case_sensitive = False

    def model_post_init(self, __context):
        # Pass branch and wipe to VectorConfig if not already set
        if self.vector is None:
            object.__setattr__(
                self,
                'vector',
                VectorConfig(storage_root=self.storage_root, branch=self.branch),
            )
        if self.icechunk is None:
            object.__setattr__(
                self,
                'icechunk',
                IcechunkConfig(
                    storage_root=self.storage_root,
                    branch=self.branch,
                ),
            )
        if self.chunking is None:
            object.__setattr__(
                self,
                'chunking',
                ChunkingConfig(),
            )
