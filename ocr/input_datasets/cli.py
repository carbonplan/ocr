"""Command-line interface for input dataset ingestion."""

import typer
from rich.panel import Panel
from rich.table import Table

from ocr.console import console
from ocr.input_datasets.base import InputDatasetConfig
from ocr.input_datasets.tensor.usfs_dillon_2023 import Dillon2023Processor
from ocr.input_datasets.tensor.usfs_riley_2025 import RileyEtAl2025Processor
from ocr.input_datasets.tensor.usfs_scott_2024 import ScottEtAl2024Processor

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
    coiled_software: str = typer.Option(..., '--software', help='Software environment to use'),
    debug: bool = typer.Option(False, '--debug', help='Enable debug logging'),
):
    """Process downloaded data and upload to S3/Icechunk."""
    if dataset not in DATASET_REGISTRY:
        console.print(f'[bold red]Error:[/] Unknown dataset: {dataset}')
        console.print(f'Available datasets: {", ".join(DATASET_REGISTRY.keys())}')
        raise typer.Exit(1)

    config = InputDatasetConfig(debug=debug)
    processor_class = DATASET_REGISTRY[dataset]['processor_class']

    # Check if processor supports Coiled
    if use_coiled and not hasattr(processor_class, 'use_coiled'):
        console.print(
            f'[bold yellow]Warning:[/] {dataset} does not support Coiled processing, ignoring --use-coiled flag'
        )
        use_coiled = False

    processor = processor_class(
        config, dry_run=dry_run, use_coiled=use_coiled, coiled_software=coiled_software
    )

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
        processor.cleanup_temp()


@app.command()
def run_all(
    dataset: str = typer.Argument(..., help='Name of the dataset to process'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Preview operations without executing'),
    use_coiled: bool = typer.Option(
        False, '--use-coiled', help='Use Coiled for distributed processing'
    ),
    debug: bool = typer.Option(False, '--debug', help='Enable debug logging'),
):
    """Run the complete pipeline: download, process, and cleanup."""
    if dataset not in DATASET_REGISTRY:
        console.print(f'[bold red]Error:[/] Unknown dataset: {dataset}')
        console.print(f'Available datasets: {", ".join(DATASET_REGISTRY.keys())}')
        raise typer.Exit(1)

    config = InputDatasetConfig(debug=debug)
    processor_class = DATASET_REGISTRY[dataset]['processor_class']

    processor = processor_class(config, dry_run=dry_run, use_coiled=use_coiled)

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
