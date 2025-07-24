# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m8g.large
# COILED --tag project=OCR


import geopandas as gpd
from tqdm import tqdm

# These are FIPS codes for states minus alaska and Hawaii.
# The census tract data is split into zipped files per state / FIPS code.
FIPS_codes = [
    '01',
    '04',
    '05',
    '06',
    '08',
    '09',
    '10',
    '11',
    '12',
    '13',
    '16',
    '17',
    '18',
    '19',
    '20',
    '21',
    '22',
    '23',
    '24',
    '25',
    '26',
    '27',
    '28',
    '29',
    '30',
    '31',
    '32',
    '33',
    '34',
    '35',
    '36',
    '37',
    '38',
    '39',
    '41',
    '42',
    '44',
    '45',
    '46',
    '47',
    '48',
    '49',
    '50',
    '51',
    '53',
    '54',
    '55',
    '56',
]

for FIPS in tqdm(FIPS_codes):
    # Using geopandas to unzip + download since fsspec allows reading zipped files over http!
    tract_url = f'https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_{FIPS}_tract.zip'
    gdf = gpd.read_file(tract_url, columns=['TRACTCE', 'GEOID', 'NAME', 'geometry'])

    gdf.to_parquet(
        f's3://carbonplan-ocr/input/fire-risk/vector/aggregated_regions/tracts/FIPS/FIPS_{FIPS}.parquet',
        compression='zstd',
        geometry_encoding='WKB',
        write_covering_bbox=True,
        schema_version='1.1.0',
    )
