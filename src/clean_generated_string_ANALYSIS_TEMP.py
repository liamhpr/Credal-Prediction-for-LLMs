import argparse
import os
import pickle
import random
import logging

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import config
import utils.utils as utils
import wandb

utils.setup_logger()
logging.info('Starting clean_generated_string.py...')

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
#parser.add_argument('--use_test_split', action='store_true')
args = parser.parse_args()

device = 'cuda'

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache


#WARNING: is the generation_tokenizer needed???

#generation_tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-350m", use_fast=False, cache_dir=config.data_dir)

wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="liam-heppner-ludwig-maximilian-university-of-munich",
    # Set the wandb project where this run will be logged.
    project="credal-prediction-for-large-language-models",
    id=args.run_id,
    config=args,
    resume='allow'
)

#wandb.init(project='credal-prediction-for-large-language-models', id=args.run_id, config=args, resume='allow')

run_name = wandb.run.name


#if args.use_test_split: 
#    path_prefix = f'{config.output_dir}sequences/{run_name}/test_split/'
#else:
#    path_prefix = f'{config.output_dir}sequences/{run_name}/train_split/'
path_prefix = f'{config.output_dir}sequences/{run_name}/'

model_path = config.get_model_path(args.generation_model)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, cache_dir=config.data_dir)

   
with open(f'{path_prefix}{args.generation_model}_ANALYSIS_TEMP_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

cleaned_sequences = []

for sample in tqdm(sequences):
    cleaned_generations = torch.ones_like(sample['generations'])
    question = sample['question']
    generated_texts = sample['generated_texts']
    cleaned_generated_texts = []

    max_len_of_generations = cleaned_generations.shape[-1]

    strings_to_filter_on = [
        '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
        'ANSWER:'
    ]

    for i, generated_text in enumerate(generated_texts):
        for string in strings_to_filter_on:
            if string in generated_text:
                generated_text = generated_text.split(string)[0] # only use the part before any filtered word
        cleaned_generated_texts.append(generated_text)

        # save the cleaned token IDs
        clean_ids = torch.cat(
            [sample['prompt'].to(device),
             torch.tensor(tokenizer(generated_text)['input_ids'][1:], device=device)])
        cleaned_generations[i, :min(len(clean_ids), max_len_of_generations)] = clean_ids[:max_len_of_generations]

    sample['cleaned_generated_texts'] = cleaned_generated_texts
    sample['cleaned_generations'] = cleaned_generations
    cleaned_sequences.append(sample)

with open(f'{path_prefix}{args.generation_model}_ANALYSIS_TEMP_generations.pkl', 'wb') as outfile:
    pickle.dump(cleaned_sequences, outfile)
