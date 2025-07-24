import coiled

batch_result = coiled.batch.run(
    command='python test_process.py', name='batch_par_test', map_over_values=['y2_x1', 'y2_x2']
)
