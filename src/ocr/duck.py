import duckdb


def duckdb_connect(*, s3_write: bool = False) -> duckdb.DuckDBPyConnection:
    """DuckDB with spatial and S3 reads (us-west-2). s3_write adds your AWS credentials."""
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'")
    if s3_write:
        con.sql('INSTALL aws; LOAD aws;')
        con.sql("CREATE SECRET (TYPE s3, PROVIDER credential_chain, REGION 'us-west-2')")
    return con
