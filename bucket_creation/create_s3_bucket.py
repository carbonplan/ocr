# Python script to create a s3 bucket and attach policies


import json

import boto3

s3_client = boto3.client('s3')
bucket_name = 'carbonplan-ocr'

response = s3_client.create_bucket(
    Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'us-west-2'}
)


# Add a bucket cors policy
s3_client.put_bucket_cors(
    Bucket=bucket_name,
    CORSConfiguration={
        'CORSRules': [
            {
                'AllowedMethods': ['GET'],
                'AllowedOrigins': ['*'],
                'AllowedHeaders': [],
                'ExposeHeaders': [],
            }
        ]
    },
)

# create empty dir structure
subfolders = [
    'input/fire-risk/tensor/',
    'input/fire-risk/vector/',
    'intermediate/',
    'output/analysis/fire-risk/tensor/',
    'output/analysis/fire-risk/vector/',
    'output/web/fire-risk/',
]

for folder in subfolders:
    s3_client.put_object(Bucket=bucket_name, Key=folder, Body='')


# 90 days for intermediate
# 1 year for input
lifecycle_config = {
    'Rules': [
        {
            'ID': 'input-rule',
            'Status': 'Enabled',
            'Filter': {'Prefix': 'input/'},
            'Expiration': {'Days': 365},
        },
        {
            'ID': 'intermediate-rule',
            'Status': 'Enabled',
            'Filter': {'Prefix': 'intermediate/'},
            'Expiration': {'Days': 90},
        },
    ]
}

s3_client.put_bucket_lifecycle_configuration(
    Bucket=bucket_name, LifecycleConfiguration=lifecycle_config
)

# setup bucket policy for get
bucket_policy = {
    'Version': '2012-10-17',
    'Statement': [
        {
            'Sid': 'PublicReadGetObject',
            'Effect': 'Allow',
            'Principal': '*',
            'Action': 's3:GetObject',
            'Resource': f'arn:aws:s3:::{bucket_name}/*',
        }
    ],
}

s3_client.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(bucket_policy))
