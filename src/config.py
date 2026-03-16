data_dir = './data/'
hf_datasets_cache = './hf_dir/hf_datasets_cache/'
hf_cache_dir = './hf_dir/hf_cache_dir/'
output_dir = './output/'

# WARNING:
# I pick T=0.5 (because Lorenz Kuhn states that 0.5 is optimal) if available, otherwise the first one.
ANALYSIS_TEMP = 0.5
#TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#TEMPERATURES = [0.490, 0.491, 0.492, 0.493, 0.494, 0.495, 0.496, 0.497, 0.498, 0.499, 0.5, 0.501, 0.502, 0.503, 0.504, 0.505]

def get_model_path(model):
    if model == 'opt-350m':
        print('using the opt-350m model')
        return '/dss/dsshome1/03/ra54sov2/Credal-Prediction-for-LLMs/src/hf_dir/hf_models/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c'
    elif model == 'opt-6.7b':
        print('using the opt-6.7b model')
        return '/dss/dsshome1/03/ra54sov2/Credal-Prediction-for-LLMs/src/hf_dir/opt-6.7b'
    elif model == 'opt-1.3b':
        print('using the opt-1.3b model')
        return '/dss/dsshome1/03/ra54sov2/Credal-Prediction-for-LLMs/src/hf_dir/opt-1.3b'
    else: 
        raise Exception(f'unknown model: {model}')
