# COILED n-tasks 1
# COILED --region us-west-2
# COILED --vm-type m8g.large
# COILED --tag project=OCR
"""Download TIGER/Line census tract shapefiles per-state -> Parquet, then aggregate.

This combines the logic from `01_tiger_zipped_to_geoparquet.py` and
`02_aggregate_tracts_to_geoparquet.py` into a single reusable CLI utility.

Steps
-----
1. For each selected state FIPS code, stream the zipped TIGER/Line tract file
   and write a geometry-preserving Parquet to S3 (one file per state).
2. After all per-state files exist, aggregate them into a single Parquet
   dataset (still a single file) using DuckDB for consistent schema + compression.

Features
--------
- Ability to (re)download only a subset of states via --states / --exclude-states.
- Skip download phase (if files already exist) and only re-aggregate via --skip-download.
- Overwrite control for both per-state outputs and the aggregated output.
- Progress logging with rich + tqdm style.
- Typer-based CLI consistent with other utilities in this repository.

Outputs
-------
S3 layout (default bucket/prefix):
  s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/tracts/
    FIPS/
      FIPS_01.parquet
      FIPS_02.parquet
      ...
    tracts.parquet  (full aggregation)

"""

from __future__ import annotations

import sys

import duckdb
import geopandas as gpd
import rich
import typer
from tqdm import tqdm

from ocr.utils import apply_s3_creds, install_load_extensions

console = rich.console.Console()
app = typer.Typer(help='Download per-state TIGER census tract data to Parquet and aggregate.')

# Default FIPS codes (all CONUS states + DC; AK + HI omitted by design)
FIPS_CODES: dict[str, str] = {
    'Alabama': '01',
    'Arizona': '04',
    'Arkansas': '05',
    'California': '06',
    'Colorado': '08',
    'Connecticut': '09',
    'Delaware': '10',
    'District of Columbia': '11',
    'Florida': '12',
    'Georgia': '13',
    'Idaho': '16',
    'Illinois': '17',
    'Indiana': '18',
    'Iowa': '19',
    'Kansas': '20',
    'Kentucky': '21',
    'Louisiana': '22',
    'Maine': '23',
    'Maryland': '24',
    'Massachusetts': '25',
    'Michigan': '26',
    'Minnesota': '27',
    'Mississippi': '28',
    'Missouri': '29',
    'Montana': '30',
    'Nebraska': '31',
    'Nevada': '32',
    'New Hampshire': '33',
    'New Jersey': '34',
    'New Mexico': '35',
    'New York': '36',
    'North Carolina': '37',
    'North Dakota': '38',
    'Ohio': '39',
    'Oklahoma': '40',
    'Oregon': '41',
    'Pennsylvania': '42',
    'Rhode Island': '44',
    'South Carolina': '45',
    'South Dakota': '46',
    'Tennessee': '47',
    'Texas': '48',
    'Utah': '49',
    'Vermont': '50',
    'Virginia': '51',
    'Washington': '53',
    'West Virginia': '54',
    'Wisconsin': '55',
    'Wyoming': '56',
}


def download_state_tracts(
    fips_code: str,
    dest_prefix: str,
    year: int = 2024,
    overwrite: bool = True,
    columns: list[str] | None = None,
) -> str:
    """Download a single state's tract geometries to Parquet on S3.

    Parameters
    ----------
    fips_code : str
        Two-digit FIPS code string.
    dest_prefix : str
        Base S3 path prefix ending with '/FIPS'. Example:
        s3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/tracts/FIPS
    year : int
        TIGER vintage (folder segment in URL)
    overwrite : bool
        If False and the object exists, skip.
    columns : list[str] | None
        Subset of columns to retain; defaults to ['TRACTCE','GEOID','NAME','geometry'].

    Returns
    -------
    str
        Destination Parquet path written.
    """
    if columns is None:
        columns = ['TRACTCE', 'GEOID', 'NAME', 'geometry']
    url = f'https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{fips_code}_tract.zip'
    dest = f'{dest_prefix}/FIPS_{fips_code}.parquet'

    # geopandas + fsspec can stream + unzip over HTTP
    console.log(f'Reading {url}')
    gdf = gpd.read_file(url, columns=columns)

    console.log(f'Writing {dest}')
    # geopandas to_parquet will use pyarrow; geometry options ensure round-trip of geometry
    try:
        gdf.to_parquet(
            dest,
            compression='zstd',  # type: ignore[arg-type]  # pyarrow supports zstd; stub is conservative
            geometry_encoding='WKB',
            write_covering_bbox=True,
            schema_version='1.1.0',
            coerce_timestamps='ms',
            allow_truncated_timestamps=True,
        )
    except Exception as e:
        if not overwrite:
            console.log(f'Skipping existing (overwrite disabled): {dest}')
            return dest
        raise e
    return dest


def aggregate_with_duckdb(
    src_glob: str,
    dest_path: str,
    overwrite: bool = True,
    install_extensions: bool = True,
):
    """Aggregate multiple per-state Parquet files into one Parquet using DuckDB."""
    if install_extensions:
        install_load_extensions()
        apply_s3_creds()

    console.log(f'Aggregating {src_glob} -> {dest_path}')
    # Use OVERWRITE_OR_IGNORE to avoid error if exists (then we can control via overwrite arg).
    if not overwrite:
        # If not overwriting, set OVERWRITE_OR_IGNORE false-ish by copying into a temp name and then skipping if exists, but duckdb syntax easiest is to check existence.
        # Simpler: if file exists and overwrite=False, skip.
        import fsspec

        fs, _, paths = fsspec.get_fs_token_paths(dest_path)
        if fs.exists(dest_path):
            console.log(
                f'Destination exists & overwrite disabled: {dest_path} -- skipping aggregation'
            )
            return

    duckdb.query(
        f"""
        COPY (
            SELECT * FROM read_parquet('{src_glob}')
        ) TO '{dest_path}' (
            FORMAT 'parquet',
            COMPRESSION 'zstd',
            OVERWRITE_OR_IGNORE true
        )
        """
    )
    console.log('Aggregation complete')


@app.command()
def main(
    output_base: str = typer.Option(
        's3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/tracts',
        help='Base S3 path for outputs (will contain FIPS/ and tracts.parquet)',
    ),
    year: int = typer.Option(2024, help='TIGER/Line vintage year'),
    states: list[str] = typer.Option(
        None, help='Subset of states to process (case-sensitive keys). If omitted, use all.'
    ),
    exclude_states: list[str] = typer.Option(
        None, help='States to exclude from processing when not using --states.'
    ),
    skip_download: bool = typer.Option(False, help='Skip per-state downloads and only aggregate'),
    overwrite_state: bool = typer.Option(True, help='Overwrite per-state Parquet files'),
    overwrite_aggregate: bool = typer.Option(True, help='Overwrite aggregated Parquet file'),
    aggregate_only_if_missing: bool = typer.Option(
        False, help='If set, aggregation runs only if destination does not already exist.'
    ),
    list_states: bool = typer.Option(False, help='List available state keys and exit'),
):
    """Run the full tract pipeline: per-state download then aggregation."""
    if list_states:
        for name, fips in FIPS_CODES.items():
            console.print(f'{name}: {fips}')
        raise typer.Exit()

    # Determine working set of states
    if states:
        unknown = [s for s in states if s not in FIPS_CODES]
        if unknown:
            console.print(f'[red]Unknown state keys: {unknown}[/red]')
            raise typer.Exit(code=1)
        selected = {k: FIPS_CODES[k] for k in states}
    else:
        selected = FIPS_CODES.copy()
        if exclude_states:
            for s in exclude_states:
                selected.pop(s, None)

    console.rule('Per-state Downloads')
    fips_prefix = f'{output_base}/FIPS'

    if not skip_download:
        for state, fips in tqdm(selected.items(), desc='States'):
            try:
                download_state_tracts(fips, fips_prefix, year=year, overwrite=overwrite_state)
            except Exception as e:
                console.print(f'[red]Failed {state} ({fips}): {e}[/red]')
                raise
    else:
        console.log('Skipping download phase as requested')

    console.rule('Aggregation')
    aggregated_path = f'{output_base}/tracts.parquet'

    if aggregate_only_if_missing:
        import fsspec

        fs, _, _ = fsspec.get_fs_token_paths(aggregated_path)
        if fs.exists(aggregated_path):
            console.log(
                f'Aggregated file exists and --aggregate-only-if-missing set: {aggregated_path} -- skipping'
            )
            raise typer.Exit()

    aggregate_with_duckdb(
        src_glob=f'{fips_prefix}/*.parquet',
        dest_path=aggregated_path,
        overwrite=overwrite_aggregate,
    )

    console.rule('Done')


if __name__ == '__main__':  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        console.print('Interrupted')
        sys.exit(1)
