# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m5.2xlarge
# COILED --tag Project=OCR


import time

import coiled
import icechunk
import rich
import typer
import xarray as xr
from distributed import Client, wait
from icechunk.xarray import to_icechunk

app = typer.Typer(help='OCR CONUS404 processing script')

console = rich.console.Console()


INPUT_ZARR_STORE_CONFIG = {
    'url': 's3://hytest/conus404/conus404_hourly.zarr',
    'storage_options': {
        'anon': True,
        'client_kwargs': {'endpoint_url': 'https://usgs.osn.mghpcc.org/'},
    },
}

DEFAULT_VARIABLES = ['Q2', 'TD2', 'PSFC', 'T2', 'V10', 'U10']
DEFAULT_SPATIAL_TILE_SIZE = 10


def setup_cluster(cluster_args: dict) -> Client:
    """Set up a Coiled cluster with the specified arguments.

    Parameters
    ----------
    cluster_args: dict
        Arguments to pass to the Coiled cluster setup.

    Returns
    -------
    Client: A Dask client connected to the Coiled cluster.
    """
    args = cluster_args
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
    """Retrieve the commit message ancestry for the specified branch.

    Parameters
    ----------
    repo: icechunk.Repository
        The icechunk repository to query.
    icechunk_branch: str
        The branch to retrieve commit messages from. Defaults to 'main'.

    Returns
    -------
    list[str]: A list of commit messages from the ancestry of the specified branch.
    """
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


def setup_repository(storage_config: dict) -> tuple:
    """Set up and return an icechunk repository.

    Parameters
    ----------
    storage_config: dict
        Configuration for the storage backend, including bucket, prefix, and region.


    Returns
    -------
    tuple: A tuple containing the icechunk repository and a writable session.
        Tuple of (repository, session)
    """
    config = storage_config
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


def load_dataset(variable: str, spatial_tile_size: int = DEFAULT_SPATIAL_TILE_SIZE) -> xr.Dataset:
    """Load the CONUS-404 dataset from S3.

    Parameters
    ----------
    variable: str
        The variable to load from the dataset.
    spatial_tile_size: int
        Size of spatial tiles for chunking.


    Returns
    -------
    xr.Dataset: An Xarray Dataset with the selected variables
    """
    vars_to_load = [variable]
    console.log(f'Loading dataset with variables: {vars_to_load}')

    ds = xr.open_zarr(
        INPUT_ZARR_STORE_CONFIG['url'], storage_options=INPUT_ZARR_STORE_CONFIG['storage_options']
    )[vars_to_load]

    console.log(f'Dataset loaded: {ds}')

    # Set chunking strategy
    ds = ds.chunk({'time': -1, 'x': spatial_tile_size, 'y': spatial_tile_size}).persist()
    start_time = time.time()
    wait(ds)
    elapsed = time.time() - start_time
    console.log(f'Dataset chunking and persisting completed in {elapsed:.2f} seconds')

    # Clean up encoding to avoid conflicts
    for variable_ in ds.variables:
        ds[variable_].encoding.pop('chunks', None)
        ds[variable_].encoding.pop('preferred_chunks', None)
        ds[variable_].encoding.pop('compressors', None)

    console.log(f'Dataset chunking configured with spatial tile size: {spatial_tile_size}')
    return ds


def process_dataset(ds: xr.Dataset, repo: icechunk.Repository):
    """
    Task to process a single job/tile and return the session with changes.

    Parameters
    ----------
    ds: xr.Dataset
        The Xarray Dataset to process.
    repo: icechunk.Repository
        The icechunk repository to write the processed dataset to.

    """
    try:
        session = repo.writable_session('main')
        start_time = time.time()
        to_icechunk(ds, session)
        session.commit('Processed dataset')
        elapsed = time.time() - start_time
        console.log(f'Job completed in {elapsed:.2f} seconds')
    except Exception as exc:
        console.log(f'Error processing dataset: {exc}')
        raise


@app.command()
def main(
    variable: str = typer.Argument(..., help='variable to process from the dataset'),
    worker_vm_type: str = typer.Option(
        'r6a.8xlarge', help='VM type for worker nodes in the Coiled cluster'
    ),
    scheduler_vm_type: str = typer.Option(
        'c6a.large', help='VM type for the scheduler node in the Coiled cluster'
    ),
    n_workers: int = typer.Option(15, help='Number of worker nodes in the Coiled cluster'),
    spatial_tile_size: int = typer.Option(
        DEFAULT_SPATIAL_TILE_SIZE, help='Size of spatial tiles for chunking'
    ),
):
    """Main entry point for processing CONUS-404 dataset."""
    console.log(f'Starting OCR CONUS-404 processing for variable: {variable}...')

    DEFAULT_CLUSTER_ARGS = {
        'name': f'ocr-conus404-hourly-osn-{variable}',
        'region': 'us-west-2',
        'n_workers': [n_workers, n_workers, n_workers],
        'tags': {'Project': 'OCR'},
        'worker_vm_types': worker_vm_type,
        'scheduler_vm_types': scheduler_vm_type,
    }

    DEFAULT_STORAGE_CONFIG = {
        'bucket': 'carbonplan-ocr',
        'prefix': f'input/conus404-hourly-icechunk/{variable}',
        'region': 'us-west-2',
    }

    client = setup_cluster(DEFAULT_CLUSTER_ARGS)

    repo, session = setup_repository(DEFAULT_STORAGE_CONFIG)

    ds = load_dataset(variable, spatial_tile_size)

    process_dataset(ds, repo)

    client.cluster.close()

    console.log(f'OCR CONUS-404 processing for variable: {variable} complete.')


if __name__ == '__main__':
    app()
