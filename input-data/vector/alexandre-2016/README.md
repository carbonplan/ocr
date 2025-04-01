# Data from: Factors related to building loss due to wildfires in the conterminous United States

- Data: https://datadryad.org/dataset/doi:10.5061/dryad.h1v2g#
- Paper: https://esajournals.onlinelibrary.wiley.com/doi/10.1002/eap.1376

To download and process the data, we provide a Python script that does the following:

1. Downloads the data from the provided URL.
2. Unarchives the downloaded file.
3. Uploads the data to a specified S3 bucket.
4. Post-process the data from CSV to Parquet format.
5. Uploads the processed data to the S3 bucket.

To run the script, make sure you have set up the local environment with `pixi`. See the contribution guide for more details on how to set up the environment.

To run the script, execute the following command in your terminal:

```bash
pixi run -e data-mgmt python input-data/vector/alexandre-2016/download_and_process.py
```
