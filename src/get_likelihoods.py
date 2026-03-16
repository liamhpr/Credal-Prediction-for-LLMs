import argparse
import os
import pickle
import random

import numpy as np
from pyarrow import output_stream
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import utils.utils as utils
import logging

import wandb

utils.setup_logger()
logging.info('Starting get_likelihoods.py...')

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation_model', type=str, default='opt-350m')
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
#parser.add_argument('--use_test_split', action='store_true')
args = parser.parse_args()

device = 'cuda'
import config

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

model_path = config.get_model_path(args.generation_model)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.float16,
                                             cache_dir=config.data_dir).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          use_fast=False,
                                          cache_dir=config.data_dir)

wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="",
    # Set the wandb project where this run will be logged.
    project="credal-prediction-for-large-language-models",
    id=args.run_id,
    config=args,
    resume='allow'
)

run_name = wandb.run.name

path_prefix = f'{config.output_dir}sequences/{run_name}/'

input_file = f'{path_prefix}{args.generation_model}_generations_similarities.pkl'
with open(input_file, 'rb') as infile:
    all_temperatures_sequences = pickle.load(infile)


def get_neg_loglikelihoods(model, sequences):

    with torch.no_grad():
        result = []
        for sample in sequences:
            result_dict = {}
            prompt = sample['prompt']
            if 'cleaned_generations' in sample:
                generations = sample['cleaned_generations'].to(device)
            else:
                generations = sample['generations'].to(device)
            id_ = sample['id']

            average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
            pointwise_mutual_information = torch.zeros((generations.shape[0],))
            sequence_embeddings = []

            for generation_index in range(generations.shape[0]):
                prompt = prompt[prompt != tokenizer.pad_token_id]
                generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id]

                # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
                target_ids = generation.clone()
                target_ids[:len(prompt)] = -100
                model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
                generation_only = generation.clone()[(len(prompt) - 1):]
                unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                   labels=generation_only,
                                                   output_hidden_states=True)
                hidden_states = model_output['hidden_states']
                average_neg_log_likelihood = model_output['loss']

                average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']
                average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
                average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood
                neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
                neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * (
                    len(generation) - len(prompt))
                pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                    generation_index] + neg_unconditioned_log_likelihoods[generation_index]

                average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
                sequence_embeddings.append(average_of_last_layer_token_embeddings)


            most_likely_generation = sample['most_likely_generation_ids'].to(device)
            target_ids = most_likely_generation.clone()
            target_ids[:len(prompt)] = -100
            model_output = model(torch.reshape(most_likely_generation, (1, -1)),
                                 labels=target_ids,
                                 output_hidden_states=True)
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood_of_most_likely_gen = model_output['loss']
            most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

            second_most_likely_generation = sample['second_most_likely_generation_ids'].to(device)
            target_ids = second_most_likely_generation.clone()
            target_ids[:len(prompt)] = -100
            model_output = model(torch.reshape(second_most_likely_generation, (1, -1)),
                                 labels=target_ids,
                                 output_hidden_states=True)
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood_of_second_most_likely_gen = model_output['loss']
            #second_most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

            neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (
                len(most_likely_generation) - len(prompt))

            sequence_embeddings = torch.stack(sequence_embeddings)
            result_dict['prompt'] = prompt
            result_dict['generations'] = generations
            result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
            result_dict['neg_log_likelihoods'] = neg_log_likelihoods
            result_dict['sequence_embeddings'] = most_likely_generation_embedding
            result_dict['most_likely_sequence_embedding'] = most_likely_generation
            result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
            result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
            result_dict['pointwise_mutual_information'] = pointwise_mutual_information
            result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen
            result_dict[
                'average_neg_log_likelihood_of_second_most_likely_gen'] = average_neg_log_likelihood_of_second_most_likely_gen
            result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen
            result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_[0]]['semantic_set_ids'], device=device)
            result_dict['id'] = id_
            result.append(result_dict)

        return result

def new_get_neg_loglikelihoods(model, samples):
    """
    Computes likelihoods for a list of samples (specific to one temperature).
    """

    with torch.no_grad():
        result = []

        # Iterate over the samples (list)
        for sample in samples:
            result_dict = {}

            prompt = sample['prompt'].to(device)
            id_ = sample['id']

            if 'cleaned_generations' in sample:
                generations = sample['cleaned_generations'].to(device)
            else:
                generations = sample['generations'].to(device)

            # Initialize storage tensors
            num_gens = generations.shape[0]
            average_neg_log_likelihoods = torch.zeros((num_gens,))
            average_unconditioned_neg_log_likelihoods = torch.zeros((num_gens,))
            neg_log_likelihoods = torch.zeros((num_gens,))
            neg_unconditioned_log_likelihoods = torch.zeros((num_gens,))
            pointwise_mutual_information = torch.zeros((num_gens,))
            sequence_embeddings = []

            # 2. Loop over the N generations for this sample
            for generation_index in range(num_gens):
                # Remove padding
                clean_prompt = prompt[prompt != tokenizer.pad_token_id]
                clean_generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id]

                # Conditional Likelihood (P(Answer | Prompt))
                target_ids = clean_generation.clone()
                target_ids[:len(clean_prompt)] = -100

                model_output = model(torch.reshape(clean_generation, (1, -1)),
                                     labels=target_ids, 
                                     output_hidden_states=True)

                average_neg_log_likelihood = model_output['loss']
                hidden_states = model_output['hidden_states']

                # unconditional likelihood (P()Answer)
                generation_only = clean_generation.clone()[(len(clean_prompt)- 1):]
                unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                   labels=generation_only,
                                                   output_hidden_states=True)

                average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']

                # Store metrics
                average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
                average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood

                # Total NLL (Average * Length)
                gen_len = len(clean_generation) - len(clean_prompt)
                neg_log_likelihoods[generation_index] = average_neg_log_likelihood * gen_len
                neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * gen_len

                pointwise_mutual_information[generation_index] = -neg_log_likelihoods[generation_index] + neg_unconditioned_log_likelihoods[generation_index]

                # Embedding (Last layer average)
                average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
                sequence_embeddings.append(average_of_last_layer_token_embeddings)

            # Stack embeddings
            sequence_embeddings = torch.stack(sequence_embeddings).squeeze(1)

            most_likely_generation = sample['most_likely_generation_ids'].to(device)

            # Mask Prompt
            target_ids = most_likely_generation.clone()
            target_ids[:len(prompt[prompt != tokenizer.pad_token_id])] = -100

            model_output = model(torch.reshape(most_likely_generation, (1, -1)),
                                 labels=target_ids,
                                 output_hidden_states=True)

            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood_of_most_likely_gen = model_output['loss']
            most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

            # Second most likely
            second_most_likely_generation = sample['second_most_likely_generation_ids'].to(device)
            target_ids = second_most_likely_generation.clone()
            target_ids[:len(prompt[prompt != tokenizer.pad_token_id])] = -100

            model_output_2 = model(torch.reshape(second_most_likely_generation, (1, -1)),
                                   labels=target_ids,
                                   output_hidden_states=True)
            average_neg_log_likelihood_of_second_most_likely_gen = model_output_2['loss']

            neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (
                len(most_likely_generation) - len(prompt[prompt != tokenizer.pad_token_id]))

            # Construct Result Dictionary
            result_dict['prompt'] = prompt.cpu()
            result_dict['generations'] = generations.cpu()

            # Metrics
            result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
            result_dict['neg_log_likelihoods'] = neg_log_likelihoods
            result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
            result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
            result_dict['pointwise_mutual_information'] = pointwise_mutual_information
            
            # Embeddings
            # FIX: In original code, this was overwritten by most_likely_generation_embedding
            result_dict['sequence_embeddings'] = sequence_embeddings.cpu() 
            result_dict['most_likely_sequence_embedding'] = most_likely_generation_embedding.cpu()
            
            # Beam Search Stats
            result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen
            result_dict['average_neg_log_likelihood_of_second_most_likely_gen'] = average_neg_log_likelihood_of_second_most_likely_gen
            result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen
            
            # Semantic IDs (Retrieved directly from sample now)
            # We rely on the previous script having populated 'semantic_set_ids'
            result_dict['semantic_set_ids'] = torch.tensor(sample['semantic_set_ids'], device=device)
            
            result_dict['id'] = id_
            result.append(result_dict)

        return result

# Main Execution
# Dictionary to store results for all temperatures
all_temperature_likelihoods = {}

# Iterate over the temperature in the input dictionary
for temp, samples in all_temperatures_sequences.items():
    logging.info(f"Computing likelihoods for Temperature {temp} ({len(samples)} samples)...")

    # Compute likelihoods for this batch 
    likelihoods_for_temp = new_get_neg_loglikelihoods(model, samples)

    # Store in the result dictionary 
    all_temperature_likelihoods[temp] = likelihoods_for_temp


# Save the final dictionary
output_filename = f'{path_prefix}{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl'
logging.info(f'Storing all likelihoods in {output_filename}')

with open(output_filename, 'wb') as outfile:
    pickle.dump(all_temperature_likelihoods, outfile)
