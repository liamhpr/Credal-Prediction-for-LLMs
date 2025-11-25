import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = os.path.join(base_dir, 'data/sets/')
hf_datasets_cache = ''
hf_cache_dir = '' # NOTE: NOT IN THE ORIGINAL FILE OF LORENZ KUHN
output_dir = os.path.join(base_dir, 'output/')
