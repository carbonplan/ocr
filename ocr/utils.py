def apply_s3_creds(region: str = 'us-west-2'):
    import boto3
    import duckdb

    session = boto3.Session()
    credentials = session.get_credentials()
    return duckdb.sql(f"""CREATE SECRET (
    TYPE s3,
    KEY_ID '{credentials.access_key}',
    SECRET '{credentials.secret_key}',
    REGION '{region}');""")


def install_load_extensions(aws: bool = True, spatial: bool = True, httpfs: bool = True):
    import duckdb

    ext_str = ''
    if aws:
        ext_str += """INSTALL aws; LOAD aws;"""
    if spatial:
        ext_str += """INSTALL SPATIAL; LOAD SPATIAL;"""
    if httpfs:
        ext_str += """INSTALL httpfs; LOAD httpfs"""
    return duckdb.sql(ext_str)
