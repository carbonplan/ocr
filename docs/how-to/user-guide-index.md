# User guides

These guides help you work with Open Climate Risk data and outputs.

```{toctree}
:maxdepth: 1

getting-started
work-with-data
100-locations-fire-risk-demo
transmission-lines-fire-risk
```

## Running the example notebooks

The notebooks in this section require the `ocr` package and its dependencies to be installed. The recommended approach is to use [pixi](https://pixi.prefix.dev/latest/installation/):

```bash
git clone https://github.com/carbonplan/ocr.git
cd ocr
pixi install
```

Then launch Jupyter inside the pixi environment:

```bash
pixi run jupyter
```

All notebooks read data directly from cloud storage (no local data download required), so the only local prerequisite is the environment itself.
