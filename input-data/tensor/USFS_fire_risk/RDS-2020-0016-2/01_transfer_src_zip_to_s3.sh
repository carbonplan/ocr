#!/bin/bash
# COILED n-tasks 1
# COILED container quay.io/carbonplan/ocr:latest
# COILED --region us-west-2
# COILED --vm-type m7a.4xlarge
# COILED --forward-aws-credentials
# COILED --tag project=OCR
# COILED --disk-size 500


# Download the zip(s)
# Should we have a async bash loop?

# bp
curl -L https://usfs-public.box.com/shared/static/7itw7p56vje2m0u3kqh91lt6kqq1i9l1.zip -o RDS-2020-0016-02-BP-CONUS.zip
unzip RDS-2020-0016-02-BP-CONUS.zip -d RDS-2020-0016-02-BP-CONUS/ && rm *.zip
s5cmd mv --sp 'RDS-2020-0016-02-BP-CONUS/BP_CONUS/BP_CONUS.tif' 's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/BP_CONUS.tif'

# CRPS
curl -L https://usfs-public.box.com/shared/static/v1wjt4r3pp0bjb05w8qfnlbm4y3m66q5.zip -o RDS-2020-0016-02-CRPS-CONUS.zip
unzip RDS-2020-0016-02-CRPS-CONUS.zip -d RDS-2020-0016-02-CRPS-CONUS/ && rm *.zip
s5cmd mv --sp 'RDS-2020-0016-02--CONUS/CRPS_CONUS/CRPS_CONUS.tif' 's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/CRPS_CONUS.tif'


# cfl
curl -L https://usfs-public.box.com/shared/static/7nb6hpw2rfc0zrhk1mv80fhbirajoqfd.zip -o RDS-2020-0016-02-CFL-CONUS.zip
unzip RDS-2020-0016-02-CFL-CONUS.zip -d RDS-2020-0016-02-CFL-CONUS/ && rm *.zip
s5cmd mv --sp 'RDS-2020-0016-02-CFL-CONUS/CFL_CONUS/CFL_CONUS.tif' 's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/CFL_CONUS.tif'

# exposure
curl -L https://usfs-public.box.com/shared/static/nbmlha1iejzzjo9y3uoehln493o2c4ad.zip -o RDS-2020-0016-02-Exposure-CONUS.zip
unzip RDS-2020-0016-02-Exposure-CONUS.zip -d RDS-2020-0016-02-Exposure-CONUS/ && rm *.zip
s5cmd mv --sp 'RDS-2020-0016-02-Exposure-CONUS/Exposure_CONUS/Exposure_CONUS.tif' 's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/Exposure_CONUS.tif'

# FLEP4
curl -L https://usfs-public.box.com/shared/static/502cm6vef6axhtljvy6b8p35tqugg2ds.zip -o RDS-2020-0016-02-FLEP4-CONUS.zip
unzip RDS-2020-0016-02-FLEP4-CONUS.zip -d RDS-2020-0016-02-FLEP4-CONUS/ && rm *.zip
s5cmd mv --sp 'RDS-2020-0016-02-FLEP4-CONUS/FLEP4_CONUS/FLEP4_CONUS.tif' 's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/FLEP4_CONUS.tif'

# FLEP8
curl -L https://usfs-public.box.com/shared/static/gwasv734wwcx77zc4wj4ntfhaxyt8mel.zip -o RDS-2020-0016-02-FLEP8-CONUS.zip
unzip RDS-2020-0016-02-FLEP8-CONUS.zip -d RDS-2020-0016-02-FLEP8-CONUS/ && rm *.zip
s5cmd mv --sp 'RDS-2020-0016-02-FLEP8-CONUS/FLEP8_CONUS/FLEP8_CONUS.tif' 's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/FLEP8_CONUS.tif'

# RPS
curl -L https://usfs-public.box.com/shared/static/88tv8byot0t22o9p1eqlrfqco3z5ouvf.zip -o RDS-2020-0016-02-RPS-CONUS.zip
unzip RDS-2020-0016-02-RPS-CONUS.zip -d RDS-2020-0016-02-RPS-CONUS/ && rm *.zip
s5cmd mv --sp 'RDS-2020-0016-02-RPS-CONUS/RPS_CONUS/RPS_CONUS.tif' 's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/RPS_CONUS.tif'

# WHP
curl -L https://usfs-public.box.com/shared/static/jz74xh0eqdezblhexwu2s2at7fqgom8n.zip -o RDS-2020-0016-02-WHP-CONUS.zip
unzip RDS-2020-0016-02-WHP-CONUS.zip -d RDS-2020-0016-02-WHP-CONUS/ && rm *.zip
s5cmd mv --sp 'RDS-2020-0016-02-WHP-CONUS/WHP_CONUS/WHP_CONUS.tif' 's3://carbonplan-ocr/input/fire-risk/tensor/USFS/RDS-2020-0016-02/input_tif/WHP_CONUS.tif'
