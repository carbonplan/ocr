import typer

from ocr.wind_spread import main as wind_spread_main

app = typer.Typer()


@app.command()
def run_wind_spread(
    bounding_box: str = typer.Argument(
        ..., help="Bounding box in the format 'xmin,ymin,xmax,ymax'"
    ),
    # fire_name: str = typer.Option(..., help='Name of the fire to process'),
    buffer: int = typer.Option(2000, help='Buffer distance in meters'),
):
    """Run wind spread analysis for a specific fire within a bounding box."""
    bbox = list(map(float, bounding_box.split(',')))
    wind_spread_main(bounding_box=bbox, buffer=buffer)


def main():
    typer.run(app())
