import argparse
import os
import pickle
import random

import numpy as np
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
parser.add_argument('--use_test_split', action='store_true')
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

model = AutoModelForCausalLM.from_pretrained(f"/dss/dsshome1/03/ra54sov2/Credal-Prediction-for-LLMs/src/hf_dir/hf_models/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c",
                                             torch_dtype=torch.float16,
                                             cache_dir=config.data_dir).cuda()
tokenizer = AutoTokenizer.from_pretrained(f"/dss/dsshome1/03/ra54sov2/Credal-Prediction-for-LLMs/src/hf_dir/hf_models/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c",
                                          use_fast=False,
                                          cache_dir=config.data_dir)

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

if args.use_test_split: 
    path_prefix = f'{config.output_dir}sequences/{run_name}/test_split/'
else:
    path_prefix = f'{config.output_dir}sequences/{run_name}/train_split/'


opt_models = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

with open(f'{path_prefix}{args.generation_model}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'{path_prefix}{args.generation_model}_generations_similarities.pkl', 'rb') as infile:
    similarities_dict = pickle.load(infile)


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


            sequence_lls = -average_neg_log_likelihoods # NOTE: Store log-likelihodds to later compute the likelihood of the clusters
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



            # NOTE: The next block is by me to compute the likelihoods of the clusters:
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            semantic_ids = torch.tensor(
                similarities_dict[id_[0]]['semantic_set_ids'])

            unique_clusters = torch.unique(semantic_ids)
            cluster_log_likelihoods = []
            cluster_sizes = []

            for c in unique_clusters:
                idx = semantic_ids == c
                ll = sequence_lls[idx]

                # log p(cluster | prompt)
                cluster_ll = torch.logsumexp(ll, dim=0)

                cluster_log_likelihoods.append(cluster_ll)
                cluster_sizes.append(idx.sum())

            cluster_log_likelihoods = torch.stack(cluster_log_likelihoods)
            cluster_sizes = torch.tensor(cluster_sizes)
            # NOTE: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
            # NOTE: These entries are for cluster log-likelihoods
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            result_dict['cluster_log_likelihoods'] = cluster_log_likelihoods
            result_dict['cluster_sizes'] = cluster_sizes
            result_dict['num_clusters'] = len(unique_clusters)

            # WARNING: Delete next two lines after testing
            print(result_dict['cluster_log_likelihoods'])
            print(torch.softmax(result_dict['cluster_log_likelihoods'], dim=0))
            # NOTE: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            result.append(result_dict)

        return result


likelihoods = get_neg_loglikelihoods(model, sequences)

print(f'storing likelihoods in {path_prefix}{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl')
with open(f'{path_prefix}{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl', 'wb') as outfile:
    pickle.dump(likelihoods, outfile)
