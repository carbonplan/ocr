# Batch Deployment

This directory contains tools for batch deployment of the OCR pipeline. We currently support two deployment methods: `coiled` and `local`. Both methods process regions in batches through the various stages of the OCR pipeline.

To run the deployment, we provide a command-line interface (CLI) that accepts parameters for region selection, platform choice, and other configuration options as shown below:

```bash
pixi run python deploy/cli.py --help


 Usage: cli.py [OPTIONS]

 Run the OCR deployment pipeline on Coiled.


╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --region-id      -r      TEXT            Region IDs to process, e.g., y10_x2 [default: None] [required]                                         │
│    --platform       -p      [coiled|local]  Platform to run the pipeline on [default: coiled]                                                      │
│    --branch         -b      [QA|prod]       Data branch path [default: QA]                                                                         │
│    --wipe           -w                      If True, wipes icechunk repo and vector data before initializing.                                      │
│    --debug          -d                      Enable Debugging Mode                                                                                  │
│    --summary-stats  -s                      Adds in spatial summary aggregations.                                                                  │
│    --help                                   Show this message and exit.                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Local (in development)

TODO

## Coiled

To run the OCR pipeline on Coiled, you need to have a Coiled account. The CLI will handle the deployment process, including setting up the necessary resources on Coiled.
To run the deployment on Coiled, use the following command:

```bash
pixi run python deploy/cli.py --region-id y10_x2 --platform coiled
```
