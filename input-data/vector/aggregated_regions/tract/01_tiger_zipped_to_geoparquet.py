# COILED n-tasks 1
# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m8g.large
# COILED --tag project=OCR


import geopandas as gpd
from tqdm import tqdm

# These are FIPS codes for states minus alaska and Hawaii.
# The census tract data is split into zipped files per state / FIPS code.
FIPS_codes = {
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


for state, FIPS in tqdm(FIPS_codes.items()):
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
