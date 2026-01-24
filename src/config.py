data_dir = './data/'
hf_datasets_cache = './hf_dir/hf_datasets_cache/'
hf_cache_dir = './hf_dir/hf_cache_dir/'
output_dir = './output/'

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
        raise(f'unknown model: {model}')
