# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m5.2xlarge
# COILED --tag Project=OCR


import time

import coiled
import dask
import icechunk
import pydantic
import rich
import typer
import xarray as xr
from icechunk.xarray import to_icechunk

app = typer.Typer(help='OCR CONUS404 processing script')

console = rich.console.Console()

DEFAULT_CLUSTER_ARGS = {
    'name': 'ocr-conus404-hourly-osn',
    'region': 'us-west-2',
    'n_workers': 50,
    'tags': {'Project': 'OCR'},
    'worker_vm_types': 'c6a.large',
    'scheduler_vm_types': 'c6a.large',
}


DEFAULT_STORAGE_CONFIG = {
    'bucket': 'carbonplan-ocr',
    'prefix': 'input/conus404-hourly-icechunk',
    'region': 'us-west-2',
}

INPUT_ZARR_STORE_CONFIG = {
    'url': 's3://hytest/conus404/conus404_hourly.zarr',
    'storage_options': {
        'anon': True,
        'client_kwargs': {'endpoint_url': 'https://usgs.osn.mghpcc.org/'},
    },
}

DEFAULT_VARIABLES = ['Q2', 'TD2', 'PSFC', 'T2', 'V10', 'U10']
DEFAULT_SPATIAL_TILE_SIZE = 8


class Job(pydantic.BaseModel):
    """Represents a processing job for a spatial tile of the dataset."""

    tile_x: int
    tile_y: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int

    @property
    def region_id(self) -> str:
        """Generate a unique identifier for this spatial region."""
        return f'x{self.x_start:04d}_{self.x_end:04d}_y{self.y_start:04d}_{self.y_end:04d}'


def setup_cluster(cluster_args: dict | None = None) -> coiled.Cluster:
    """Set up a Coiled cluster with the specified arguments."""
    args = cluster_args or DEFAULT_CLUSTER_ARGS
    console.log(f'Setting up Coiled cluster with args: {args}')
    try:
        cluster = coiled.Cluster(**args)
        client = cluster.get_client()
        console.log(
            f'Cluster {cluster.name} created with {len(client.scheduler_info()["workers"])} workers.'
        )
        return client
    except Exception as exc:
        console.log(f'Error setting up cluster: {exc}')
        raise


def get_commit_messages_ancestry(
    repo: icechunk.Repository, icechunk_branch: str = 'main'
) -> list[str]:
    """Retrieve the commit message ancestry for the specified branch."""
    hist = repo.ancestry(branch=icechunk_branch)
    commit_messages = []

    for ancestor in hist:
        commit_messages.append(ancestor.message)

    # Separate commits by ',' and handle case of single length ancestry commit history
    split_commits = [
        msg
        for message in commit_messages
        for msg in (message.split(',') if ',' in message else [message])
    ]

    return split_commits


def setup_repository(storage_config: dict | None = None) -> tuple:
    """Set up and return an icechunk repository.



    Returns
    -------
    tuple: A tuple containing the icechunk repository and a writable session.
        Tuple of (repository, session)
    """
    config = storage_config or DEFAULT_STORAGE_CONFIG
    console.log(f'Setting up icechunk repository with config: {config}')

    storage = icechunk.s3_storage(
        bucket=config['bucket'],
        prefix=config['prefix'],
        region=config['region'],
    )

    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session('main')

    console.log(f'Repository setup complete: {config["bucket"]}/{config["prefix"]}')
    return repo, session


def load_dataset(
    variables: list[str] | None = None, spatial_tile_size: int = DEFAULT_SPATIAL_TILE_SIZE
) -> xr.Dataset:
    """Load the CONUS-404 dataset from S3.

    Parameters:
    ----------
        variables: List of variables to load
        spatial_tile_size: Size of spatial tiles for chunking

    Returns:
        Xarray Dataset with the selected variables
    """
    vars_to_load = variables or DEFAULT_VARIABLES
    console.log(f'Loading dataset with variables: {vars_to_load}')

    ds = xr.open_zarr(
        INPUT_ZARR_STORE_CONFIG['url'], storage_options=INPUT_ZARR_STORE_CONFIG['storage_options']
    )[vars_to_load]

    console.log(f'Dataset loaded: {ds}')

    # Set chunking strategy
    ds = ds.chunk({'time': -1, 'x': spatial_tile_size, 'y': spatial_tile_size})

    # Clean up encoding to avoid conflicts
    for variable in ds.variables:
        ds[variable].encoding.pop('chunks', None)
        ds[variable].encoding.pop('preferred_chunks', None)
        ds[variable].encoding.pop('compressors', None)

    console.log(f'Dataset chunking configured with spatial tile size: {spatial_tile_size}')
    return ds


def initialize_store_if_needed(repo, session, ds):
    """Initialize the icechunk store if it hasn't been done yet.

    Parameters:
    ----------
    repo: The icechunk repository
    session: The writable session
    ds: The dataset to initialize with
    """
    if 'initialize store' not in get_commit_messages_ancestry(repo):
        console.log('Initializing template store')
        ds.to_zarr(session.store, compute=False, mode='w', consolidated=False)
        session.commit('initialize store')
        console.log('Store initialization complete')
    else:
        console.log('Store already initialized, skipping')


def generate_jobs(
    ds: xr.Dataset, repo: icechunk.Repository, spatial_tile_size: int = DEFAULT_SPATIAL_TILE_SIZE
) -> list[Job]:
    """Generate processing jobs for each spatial tile.

    Parameters:
    ----------
    ds: The dataset to process
    repo: The icechunk repository for checking completed jobs
    spatial_tile_size: Size of spatial tiles

    Returns
    -------
    list[Job]: A List of Job objects for tiles that need processing
    """
    x_size = ds.sizes['x']
    y_size = ds.sizes['y']
    x_tiles = x_size // spatial_tile_size
    y_tiles = y_size // spatial_tile_size

    console.log(f'Generating jobs for {x_tiles}x{y_tiles} tiles')

    # Get list of already completed regions
    completed = get_commit_messages_ancestry(repo)

    # Prepare jobs for remaining tiles
    jobs = []
    for tile_x in range(x_tiles):
        for tile_y in range(y_tiles):
            x_start = tile_x * spatial_tile_size
            y_start = tile_y * spatial_tile_size
            x_end = x_start + spatial_tile_size
            y_end = y_start + spatial_tile_size

            job = Job(
                tile_x=tile_x,
                tile_y=tile_y,
                x_start=x_start,
                x_end=x_end,
                y_start=y_start,
                y_end=y_end,
            )

            if job.region_id in completed:
                console.log(f'Tile {job.region_id} already processed; skipping.')
                continue

            jobs.append(job)

    console.log(f'Generated {len(jobs)} jobs to process out of {x_tiles * y_tiles} total tiles')
    return jobs


def process_jobs(ds: xr.Dataset, jobs: list[Job], repo: icechunk.Repository):
    """Process all jobs, writing each tile to the icechunk repository.

    Parameters:
    -----------
    ds: The dataset to process
    jobs: List of jobs to process
    session: The writable session for the repository
    """
    total_jobs = len(jobs)
    console.log(f'Starting processing of {total_jobs} jobs')

    for i, job in enumerate(jobs, 1):
        session = repo.writable_session('main')
        start_time = time.time()
        console.log(f'Processing job {i}/{total_jobs}: {job.region_id}')

        try:
            dset = dask.optimize(
                ds.isel(x=slice(job.x_start, job.x_end), y=slice(job.y_start, job.y_end))
            )[0]

            to_icechunk(dset, session, region='auto')
            session.commit(job.region_id)

            elapsed = time.time() - start_time
            console.log(f'Job {i}/{total_jobs} completed in {elapsed:.2f} seconds: {job.region_id}')

        except Exception as e:
            console.log(f'Error processing job {job.region_id}: {e}')
            continue


@app.command()
def main(
    variables: list[str] = typer.Option(
        DEFAULT_VARIABLES, help='List of variables to process from the dataset'
    ),
    spatial_tile_size: int = typer.Option(
        DEFAULT_SPATIAL_TILE_SIZE, help='Size of spatial tiles for chunking'
    ),
):
    """Main entry point for processing CONUS-404 dataset."""
    console.log('Starting OCR CONUS-404 processing...')

    client = setup_cluster()

    repo, session = setup_repository()

    ds = load_dataset(variables, spatial_tile_size)

    initialize_store_if_needed(repo, session, ds)

    jobs = generate_jobs(ds, repo, spatial_tile_size)

    process_jobs(ds, jobs, repo)

    client.cluster.close()

    console.log('OCR CONUS-404 processing complete.')


if __name__ == '__main__':
    app()
