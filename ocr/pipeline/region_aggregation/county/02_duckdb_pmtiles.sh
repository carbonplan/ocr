
#!/bin/bash

# COILED container quay.io/carbonplan/ocr:latest
# COILED n-tasks 1
# COILED region us-west-2
# COILED --forward-aws-credentials
# COILED --tag project=OCR
# COILED --vm-type c7a.xlarge
# COILED --name tiles_duckdb_test


duckdb -c "
load spatial;

COPY (
    SELECT
        'Feature' AS type,
        json_object(
            'county_name', county_name,
            'building_count', building_count,
            'avg_risk_2011_horizon_1', avg_risk_2011_horizon_1,
            'avg_risk_2011_horizon_15', avg_risk_2011_horizon_15,
            'avg_risk_2011_horizon_30', avg_risk_2011_horizon_30,
            'avg_risk_2047_horizon_1', avg_risk_2047_horizon_1,
            'avg_risk_2047_horizon_15', avg_risk_2047_horizon_15,
            'avg_risk_2047_horizon_30', avg_risk_2047_horizon_30,
            'avg_wind_risk_2011_horizon_1', avg_wind_risk_2011_horizon_1,
            'avg_wind_risk_2011_horizon_15', avg_wind_risk_2011_horizon_15,
            'avg_wind_risk_2011_horizon_30', avg_wind_risk_2011_horizon_30,
            'avg_wind_risk_2047_horizon_1', avg_wind_risk_2047_horizon_1,
            'avg_wind_risk_2047_horizon_15', avg_wind_risk_2047_horizon_15,
            'avg_wind_risk_2047_horizon_15', avg_wind_risk_2047_horizon_15,
            'risk_2011_horizon_1', risk_2011_horizon_1,
            'risk_2011_horizon_15', risk_2011_horizon_15,
            'risk_2011_horizon_30', risk_2011_horizon_30,
            'risk_2047_horizon_1', risk_2047_horizon_1,
            'risk_2047_horizon_15', risk_2047_horizon_15,
            'risk_2047_horizon_30', risk_2047_horizon_30,
            'wind_risk_2011_horizon_1', wind_risk_2011_horizon_1,
            'wind_risk_2011_horizon_15', wind_risk_2011_horizon_15,
            'wind_risk_2011_horizon_30', wind_risk_2011_horizon_30,
            'wind_risk_2047_horizon_1', wind_risk_2047_horizon_1,
            'wind_risk_2047_horizon_15', wind_risk_2047_horizon_15,
            'wind_risk_2047_horizon_30', wind_risk_2047_horizon_30
             ) AS properties,
        json(ST_AsGeoJson(geometry)) AS geometry

    FROM read_parquet('s3://carbonplan-ocr/intermediate/fire-risk/vector/county/county_summary_stats.parquet')
) TO STDOUT (FORMAT json);" | tippecanoe -o aggregated.pmtiles -l risk -n "county" -f -P --drop-smallest-as-needed -q --extend-zooms-if-still-dropping -zg



echo tippecanoe tiles done

s5cmd cp --sp "aggregated.pmtiles" "s3://carbonplan-ocr/intermediate/fire-risk/vector/QA/counties.pmtiles"
