# COILED --region us-west-2
# COILED --forward-aws-credentials
# COILED --vm-type m8g.medium
# COILED --scheduler-vm-type m8g.medium
# COILED --tag project=OCR


import os

filename = os.environ['COILED_BATCH_TASK_INPUT']
print(filename)
