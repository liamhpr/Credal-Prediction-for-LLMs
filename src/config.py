data_dir = ''
hf_datasets_cache = ''
hf_cache_dir = ''
output_dir = ''

# WARNING:
# We pick T=0.5 (because Lorenz Kuhn states that 0.5 is optimal). 
ANALYSIS_TEMP = 0.5

def get_model_path(model):
    if model == 'opt-350m':
        print('using the opt-350m model')
        return ''
    elif model == 'opt-6.7b':
        print('using the opt-6.7b model')
        return ''
    elif model == 'opt-1.3b':
        print('using the opt-1.3b model')
        return ''
    else: 
        raise Exception(f'unknown model: {model}')
