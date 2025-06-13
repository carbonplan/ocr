# Data Pipeline

![](../assets/ocr_data_flow.png)

## Usage

`ocr/main.py` is a click CLI that can be used to dispatch jobs on coiled.
For example running: `uv run python main.py -c -r y5_x2 -r y5_x3`, would dispatch two coiled batch jobs. One for each region_id specified.
