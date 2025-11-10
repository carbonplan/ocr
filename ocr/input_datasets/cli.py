"""Command-line interface for input dataset ingestion."""

import typer
from rich.panel import Panel
from rich.table import Table

from ocr.console import console
from ocr.input_datasets.base import InputDatasetConfig
from ocr.input_datasets.tensor.calfire_fhsz import CalfireFHSZProcessor
from ocr.input_datasets.tensor.usfs_dillon_2023 import Dillon2023Processor
from ocr.input_datasets.tensor.usfs_riley_2025 import RileyEtAl2025Processor
from ocr.input_datasets.tensor.usfs_scott_2024 import ScottEtAl2024Processor
from ocr.input_datasets.vector.census_tiger import CensusTigerProcessor
from ocr.input_datasets.vector.overture import OvertureProcessor

app = typer.Typer(help='Ingest and process input datasets for OCR')


# Registry of available datasets
DATASET_REGISTRY = {
    'scott-et-al-2024': {
        'processor_class': ScottEtAl2024Processor,
        'type': 'tensor',
        'description': 'USFS Wildfire Risk to Communities (2nd Edition, RDS-2020-0016-02)',
    },
    'riley-et-al-2025': {
        'processor_class': RileyEtAl2025Processor,
        'type': 'tensor',
        'description': 'USFS Probabilistic Wildfire Risk - 2011 & 2047 Climate Runs (RDS-2025-0006)',
    },
    'dillon-et-al-2023': {
        'processor_class': Dillon2023Processor,
        'type': 'tensor',
        'description': 'USFS Spatial datasets of probabilistic wildfire risk components for the United States (270m) (3rd Edition) (RDS-2016-0034-3)',
    },
    'overture-maps': {
        'processor_class': OvertureProcessor,
        'type': 'vector',
        'description': 'Overture Maps building and address data for CONUS (release 2025-09-24.0)',
    },
    'census-tiger': {
        'processor_class': CensusTigerProcessor,
        'type': 'vector',
        'description': 'US Census TIGER/Line geographic boundaries (blocks, tracts, counties)',
    },
    'calfire-fhsz': {
        'processor_class': CalfireFHSZProcessor,
        'type': 'tensor',
        'description': 'California Fire Hazard Severity Zones (FHSZ)',
    },
}


@app.command()
def list_datasets():
    """List all available datasets that can be ingested."""
    table = Table(title='Available Input Datasets', show_header=True, header_style='bold magenta')
    table.add_column('Dataset Name', style='cyan', no_wrap=True)
    table.add_column('Type', style='green')
    table.add_column('Description', style='white')

    for name, info in sorted(DATASET_REGISTRY.items()):
        table.add_row(name, info['type'], info['description'])

    console.print(table)


@app.command()
def download(
    dataset: str = typer.Argument(..., help='Name of the dataset to download'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Preview operations without executing'),
    debug: bool = typer.Option(False, '--debug', help='Enable debug logging'),
):
    """Download raw source data for a dataset."""
    if dataset not in DATASET_REGISTRY:
        console.print(f'[bold red]Error:[/] Unknown dataset: {dataset}')
        console.print(f'Available datasets: {", ".join(DATASET_REGISTRY.keys())}')
        raise typer.Exit(1)

    config = InputDatasetConfig(debug=debug)
    processor_class = DATASET_REGISTRY[dataset]['processor_class']
    processor = processor_class(config, dry_run=dry_run)

    console.print(
        Panel(
            f'[bold]Dataset:[/] {dataset}\n'
            f'[bold]Description:[/] {DATASET_REGISTRY[dataset]["description"]}\n'
            f'[bold]Dry Run:[/] {dry_run}',
            title='Download Configuration',
            border_style='cyan',
        )
    )

    try:
        processor.download()
        console.print(f'[bold green]✓[/] Download completed for {dataset}')
    except Exception as e:
        console.print(f'[bold red]✗[/] Download failed: {e}')
        raise typer.Exit(1)


@app.command()
def process(
    dataset: str = typer.Argument(..., help='Name of the dataset to process'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Preview operations without executing'),
    use_coiled: bool = typer.Option(
        False, '--use-coiled', help='Use Coiled for distributed processing'
    ),
    coiled_software: str | None = typer.Option(
        None, '--software', help='Software environment to use (required if --use-coiled is set)'
    ),
    debug: bool = typer.Option(False, '--debug', help='Enable debug logging'),
    overture_data_type: str = typer.Option(
        'both',
        '--overture-data-type',
        help='For overture-maps: which data to process (buildings, addresses, or both)',
    ),
    census_geography_type: str = typer.Option(
        'all',
        '--census-geography-type',
        help='For census-tiger: which geography to process (blocks, tracts, counties, or all)',
    ),
    census_subset_states: list[str] = typer.Option(
        None,
        '--census-subset-states',
        help='For census-tiger: subset of states to process (e.g., California Oregon)',
    ),
):
    """Process downloaded data and upload to S3/Icechunk."""
    if dataset not in DATASET_REGISTRY:
        console.print(f'[bold red]Error:[/] Unknown dataset: {dataset}')
        console.print(f'Available datasets: {", ".join(DATASET_REGISTRY.keys())}')
        raise typer.Exit(1)

    config = InputDatasetConfig(debug=debug)
    processor_class = DATASET_REGISTRY[dataset]['processor_class']

    # Build processor kwargs
    processor_kwargs = {'config': config, 'dry_run': dry_run}

    # Add dataset-specific parameters
    if dataset == 'overture-maps':
        processor_kwargs['data_type'] = overture_data_type
    elif dataset == 'census-tiger':
        processor_kwargs['geography_type'] = census_geography_type
        processor_kwargs['subset_states'] = census_subset_states

    processor_kwargs['use_coiled'] = use_coiled
    processor_kwargs['coiled_software'] = coiled_software

    processor = processor_class(**processor_kwargs)

    console.print(
        Panel(
            f'[bold]Dataset:[/] {dataset}\n'
            f'[bold]Description:[/] {DATASET_REGISTRY[dataset]["description"]}\n'
            f'[bold]Dry Run:[/] {dry_run}\n'
            f'[bold]Use Coiled:[/] {use_coiled}',
            title='Processing Configuration',
            border_style='cyan',
        )
    )

    try:
        processor.process()
        console.print(f'[bold green]✓[/] Processing completed for {dataset}')
    except Exception as e:
        console.print(f'[bold red]✗[/] Processing failed: {e}')
        raise typer.Exit(1)
    finally:
        console.log('Exiting processing command.')


@app.command()
def run_all(
    dataset: str = typer.Argument(..., help='Name of the dataset to process'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Preview operations without executing'),
    use_coiled: bool = typer.Option(
        False, '--use-coiled', help='Use Coiled for distributed processing'
    ),
    debug: bool = typer.Option(False, '--debug', help='Enable debug logging'),
    overture_data_type: str = typer.Option(
        'both',
        '--overture-data-type',
        help='For overture-maps: which data to process (buildings, addresses, or both)',
    ),
    census_geography_type: str = typer.Option(
        'all',
        '--census-geography-type',
        help='For census-tiger: which geography to process (blocks, tracts, counties, or all)',
    ),
    census_subset_states: list[str] = typer.Option(
        None,
        '--census-subset-states',
        help='For census-tiger: subset of states to process (e.g., California Oregon)',
    ),
):
    """Run the complete pipeline: download, process, and cleanup."""
    if dataset not in DATASET_REGISTRY:
        console.print(f'[bold red]Error:[/] Unknown dataset: {dataset}')
        console.print(f'Available datasets: {", ".join(DATASET_REGISTRY.keys())}')
        raise typer.Exit(1)

    config = InputDatasetConfig(debug=debug)
    processor_class = DATASET_REGISTRY[dataset]['processor_class']

    processor_kwargs = {'config': config, 'dry_run': dry_run}

    # Add dataset-specific parameters
    if dataset == 'overture-maps':
        processor_kwargs['data_type'] = overture_data_type
    elif dataset == 'census-tiger':
        processor_kwargs['geography_type'] = census_geography_type
        processor_kwargs['subset_states'] = census_subset_states

    processor_kwargs['use_coiled'] = use_coiled

    processor = processor_class(**processor_kwargs)

    console.print(
        Panel(
            f'[bold]Dataset:[/] {dataset}\n'
            f'[bold]Description:[/] {DATASET_REGISTRY[dataset]["description"]}\n'
            f'[bold]Dry Run:[/] {dry_run}\n'
            f'[bold]Use Coiled:[/] {use_coiled}',
            title='Pipeline Configuration',
            border_style='cyan',
        )
    )

    try:
        processor.run_all()
        console.print(f'[bold green]✓[/] Complete pipeline finished for {dataset}')
    except Exception as e:
        console.print(f'[bold red]✗[/] Pipeline failed: {e}')
        raise typer.Exit(1)


if __name__ == '__main__':
    app()
