import argparse
import os
import pickle
import random

import config
import numpy as np
import torch
import wandb
import utils.utils as utils 
import logging
import scipy.optimize
from joblib import Parallel, delayed

utils.setup_logger()
logging.info('Starting compute_confidence_measure.py...')

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--evaluation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--verbose', type=bool, default=True)
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

wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="liam-heppner-ludwig-maximilian-university-of-munich",
    # Set the wandb project where this run will be logged.
    project="credal-prediction-for-large-language-models",
    id=args.run_id,
    config=args,
    resume='allow'
)
#wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')

run_name = wandb.run.name

llh_shift = torch.tensor(5.0)


def get_overall_log_likelihoods(list_of_results):
    """Compute log likelihood of all generations under their given context.
    
    list_of_results: list of dictionaries with keys:
    
    returns: dictionary with keys: 'neg_log_likelihoods', 'average_neg_log_likelihoods'
             that contains tensors of shape (num_models, num_generations, num_samples_per_generation)
    """

    result_dict = {}

    list_of_keys = ['neg_log_likelihoods', 'average_neg_log_likelihoods', 'sequence_embeddings',\
                    'pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                    'neg_log_likelihood_of_most_likely_gen', 'semantic_set_ids']

    for key in list_of_keys:
        list_of_ids = []
        overall_results = []
        for model_size, result in list_of_results:
            results_per_model = []
            for sample in result:
                average_neg_log_likelihoods = sample[key]
                list_of_ids.append(sample['id'][0])
                results_per_model.append(average_neg_log_likelihoods)

            results_per_model = torch.stack(results_per_model)

            overall_results.append(results_per_model)

        if key != 'sequence_embeddings':
            overall_results = torch.stack(overall_results)

        result_dict[key] = overall_results

    result_dict['ids'] = list_of_ids
    return result_dict


def get_mutual_information(log_likelihoods):
    """Compute confidence measure for a given set of likelihoods"""

    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    tiled_mean = mean_across_models.tile(log_likelihoods.shape[0], 1, 1)
    diff_term = torch.exp(log_likelihoods) * log_likelihoods - torch.exp(tiled_mean) * tiled_mean
    f_j = torch.div(torch.sum(diff_term, dim=0), diff_term.shape[0])
    mutual_information = torch.div(torch.sum(torch.div(f_j, mean_across_models), dim=1), f_j.shape[-1])

    return mutual_information


def get_log_likelihood_variance(neg_log_likelihoods):
    """Compute log likelihood variance of approximate posterior predictive"""
    mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
    variance_of_neg_log_likelihoods = torch.var(mean_across_models, dim=1)

    return variance_of_neg_log_likelihoods


def get_log_likelihood_mean(neg_log_likelihoods):
    """Compute softmax variance of approximate posterior predictive"""
    mean_across_models = torch.mean(neg_log_likelihoods, dim=0)
    mean_of_neg_log_likelihoods = torch.mean(mean_across_models, dim=1)

    return mean_of_neg_log_likelihoods


def get_mean_of_poinwise_mutual_information(pointwise_mutual_information):
    """Compute mean of pointwise mutual information"""
    mean_across_models = torch.mean(pointwise_mutual_information, dim=0)
    return torch.mean(mean_across_models, dim=1)


def get_predictive_entropy(log_likelihoods):
    """Compute predictive entropy of approximate posterior predictive"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    entropy = -torch.sum(mean_across_models, dim=1) / torch.tensor(mean_across_models.shape[1])
    return entropy


def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids):
    """Compute the semantic entropy"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    # This is ok because all the models have the same semantic set ids
    semantic_set_ids = semantic_set_ids[0]
    entropies = []
    for row_index in range(mean_across_models.shape[0]): # for each question do:
        aggregated_likelihoods = [] # to store the likelihoods for each cluster
        row = mean_across_models[row_index]
        semantic_set_ids_row = semantic_set_ids[row_index]
        for semantic_set_id in torch.unique(semantic_set_ids_row): # for each cluster do:
            aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0)) # compute and append each cluster likelihood
        aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift # create a tensor from the cluster likelihoods
        entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0]) # compute entropy
        entropies.append(entropy) # append entropy (entropy over the answers for one question)

    return torch.tensor(entropies)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>                                                                    <<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>                         Credal Entropy                             <<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>                                                                    <<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def entropy(p):
    """Helper: Shannon Entropy in nats (base e). Handles 0 log 0"""
    # Add a tiny epsilon to avoid log(0), or use specialized func
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p*np.log(p))

def neg_entropy(p):
    """Helper: Negative Entropy (for minimization to achieve MaxEnt)"""
    return -entropy(p)

def solve_credal_entropy(bounds):
    """
    Computes Lower and Upper Shannon Entropy for a single credal set defined by bounds.
    Bounds: (K, 2) array where col 0 is Lower, col 1 is Upper 
    """
    K = bounds.shape[0]
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    # 1. Sanity Check: Is the set empty?
    # sum(lower) must be <= 1 and sum(upper) must be >= 1
    assert (np.sum(lower_bounds) > 1.0 + 1e-6 or np.sum(upper_bounds) < 1.0 - 1e-6)

    x0 = (lower_bounds + upper_bounds)/ 2.0
    x0 = x0 / np.sum(x0)

    # Constraints: sum(p) = 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

    #Box bounds for optimization
    scipy_bounds = [(l, u) for l, u in bounds]

    # --- A. Compute UPPER Entropy (Maximize H -> Minimize -H) ---
    # This is a Convex Optimization Problem (easy/reliable for SLSQP)
    res_max = scipy.optimize.minimize(
        neg_entropy, 
        x0,
        method='SLSQP',
        bounds=scipy_bounds,
        constraints=constraints
    )

    ue = -res_max.fun if res_max.success else entropy(x0)

    # --- B. Compute LOWER Entropy (Minimize H) ---
    # Minimizing a concave function (entropy) is mathematically hard (non-convex).
    # The minimum always lies at a vertex of the polytope.
    # Heuristic for 2022 stack: Try to concentrate mass on one class (k) 
    # as much as possible to minimize uncertainty.
    best_le = np.inf

    # Strategy: For each class, try to maximize its prob (make it "certain")
    # This approximates finding the vertex with lowest entropy.
    for k in range(K):
        # Construct a "spiky" guess centered on class k
        # Set k to its max, others to min, then normalize
        p_guess = lower_bounds.copy()

        # How much slack do we have to distribute?
        current_sum = np.sum(p_guess)
        slack = 1.0 - current_sum

        # Add as much slack as possible to class k
        can_add = upper_bounds[k] - p_guess[k]
        add_amount = min(slack, can_add)
        p_guess[k] += add_amount

        # If there is still slack, distribute it to others (greedy)
        # (This part is simple distribution to ensure sum=1 for the solver start)
        slack = 1.0 - np.sum(p_guess)
        if slack > 1e-9:
            # Distribute remaining slack to whoever has room
            for j in range(K):
                can_add = upper_bounds[j] - p_guess[j]
                amt = min(slack, can_add)
                p_guess[j] += amt
                slack -= amt
                if slack < 1e-9: break

        # Run optimization starting from this "spiky" guess
        res_min = scipy.optimize.minimize(
            entropy, # Minimize H directly
            p_guess, 
            method='SLSQP',
            bounds=scipy_bounds,
            constraints=constraints
        )

        le_candidate = res_min.fun if res_min.success else entropy(p_guess)
        if le_candidate < best_le:
            best_le = le_candidate

    return best_le, ue



def batched_entropy_diff(intervals, batch_size=128, n_jobs=-1):
    """ 
    Computes (Upper Entropy - Lower Entropy) for a batch of Credal intervals.
    Expects intervals of shape (N_samples, N_classes, 2).
    """
    n_instances = intervals.shape[0]

    # Use Joblib to parallelize the expensive scipy optimization
    # This replaces the batch loop with parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(solve_credal_entropy)(intervals[i])
        for i in range(n_instances)
    )

    results = np.array(results) # Shape (N, 2)

    # results[:, 0] is Lower Entropy
    # results[:, 1] is Upper Entropy
    diffs = results[:, 1] - results[:, 0]

    return diffs


    """
    # Process in batches to save memory/compute
    for start in range(0, n_instances, batch_size):
        end = min(start + batch_size, n_instances)
        batch = intervals[start:end]

        # NOTE:the upper and lower shannon entropy might be calculated differently depending on the version:
        # probly expects the bounds. Depending on version, it might take 
        # (N, K, 2) or separate (N, K) arrays. 
        # Assuming standard usage of passing the interval structure:
        ue = upper_entropy(batch, n_jobs=n_jobs)
        le = upper_entropy(batch, n_jobs=n_jobs)

        results.append(ue - le)

    return np.concatenate(results, axis=0)
    """

def get_credal_entropy_over_concepts(log_likelihoods, semantic_set_ids): 
    """Compute EU (epistemic uncertainty) by computing upper and lower Shannon Entropy over the Credal set"""
    M = log_likelihoods.shape[0]
    logM = torch.log(torch.tensor(M, dtype=log_likelihoods.dtype, device=log_likelihoods.device))

    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - logM

    semantic_set_ids = semantic_set_ids[0]
    all_upperbounds = []
    all_lowerbounds =[]

    for row_index in range(mean_across_models.shape[0]): # for each question do:
        row_log_probs = mean_across_models[row_index]
        semantic_set_ids_row = semantic_set_ids[row_index]

        aggregated_cluster_likelihoods = []
        deltas = []

        # NOTE: Should I normalize the sequence probabilities to compute the cluster lls and lower and upper bounds???
        # normalized_sequence_probs = torch.nn.functional.softmax(row_log_probs, dim=0)

        for semantic_set_id in torch.unique(semantic_set_ids_row): # for each cluster do:
            # compute cluster log-likelihood  
            sequence_probs = torch.exp(row_log_probs.to(semantic_set_ids_row.device)[semantic_set_ids_row == semantic_set_id])
            max_p = torch.max(sequence_probs ) # find highest sequence probability
            min_p = torch.min(sequence_probs ) # find lowest sequence probability
            deltas.append(max_p - min_p) # compute difference between highest and lowest probability  

            aggregated_cluster_likelihood = torch.logsumexp(row_log_probs[semantic_set_ids_row == semantic_set_id], dim=0) # compute cluster likelihood
            aggregated_cluster_likelihoods.append(aggregated_cluster_likelihood) # store cluster likelihood

        aggregated_cluster_likelihoods = torch.stack(aggregated_cluster_likelihoods) # store cluster likelihoods as tensor
        current_upperbounds = torch.nn.functional.softmax(aggregated_cluster_likelihoods, dim=0) # apply softmax to cluster likelihoods
        all_upperbounds.append(current_upperbounds)

        deltas = torch.stack(deltas) # store differences between upper and lower bounds as tensor
        all_lowerbounds.append(current_upperbounds - deltas) # compute lower bounds

    # NOTE: This next part is the computation of the entropy diff (epistemic uncertainty)

    # Convert to numpy for Probly
    # Stack to shape (N_samples, N_classes)
    np_lowerbounds = torch.stack(all_lowerbounds).detach().cpu().numpy()
    np_upperbounds = torch.stack(all_upperbounds).detach().cpu().numpy()

    # Combine into shape (N_samples, N_classes, 2)
    # credal_intervals[i, j, 0] is lower bound
    # credal_intervals[i, j, 1] is upper bound
    credal_intervals = np.stack([np_lowerbounds, np_upperbounds], axis=-1)

    entropy_diffs_np = batched_entropy_diff(credal_intervals)

    return torch.from_numpy(entropy_diffs_np).to(log_likelihoods.device)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>                                                                    <<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def get_margin_probability_uncertainty_measure(log_likelihoods):
    """Compute margin probability uncertainty measure"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    topk_likelihoods, indices = torch.topk(mean_across_models, 2, dim=1, sorted=True)
    margin_probabilities = np.exp(topk_likelihoods[:, 0]) - np.exp(topk_likelihoods[:, 1])

    return margin_probabilities


list_of_results = []

with open(f'{config.output_dir}sequences/{run_name}/{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl',
          'rb') as infile:
    sequences = pickle.load(infile)
    list_of_results.append((args.evaluation_model, sequences))

overall_results = get_overall_log_likelihoods(list_of_results)
mutual_information = get_mutual_information(-overall_results['neg_log_likelihoods'])
predictive_entropy = get_predictive_entropy(-overall_results['neg_log_likelihoods'])
predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods'],
                                                                        overall_results['semantic_set_ids'])
unnormalised_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['neg_log_likelihoods'],
                                                                          overall_results['semantic_set_ids'])
credal_entropy_over_concepts = get_credal_entropy_over_concepts(-overall_results['average_neg_log_likelihoods'], 
                                                                        overall_results['semantic_set_ids'])

margin_measures = get_margin_probability_uncertainty_measure(-overall_results['average_neg_log_likelihoods'])
unnormalised_margin_measures = get_margin_probability_uncertainty_measure(-overall_results['neg_log_likelihoods'])


def get_number_of_unique_elements_per_row(tensor):
    assert len(tensor.shape) == 2
    return torch.count_nonzero(torch.sum(torch.nn.functional.one_hot(tensor), dim=1), dim=1)


number_of_semantic_sets = get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][0])
average_predictive_entropy = get_predictive_entropy(-overall_results['average_neg_log_likelihoods'])
average_predictive_entropy_on_subsets = []
predictive_entropy_on_subsets = []
semantic_predictive_entropy_on_subsets = []
num_predictions = overall_results['average_neg_log_likelihoods'].shape[-1]
number_of_semantic_sets_on_subsets = []
for i in range(1, num_predictions + 1):
    offset = num_predictions * (i / 100)
    average_predictive_entropy_on_subsets.append(
        get_predictive_entropy(-overall_results['average_neg_log_likelihoods'][:, :, :int(i)]))
    predictive_entropy_on_subsets.append(get_predictive_entropy(-overall_results['neg_log_likelihoods'][:, :, :int(i)]))
    semantic_predictive_entropy_on_subsets.append(
        get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods'][:, :, :int(i)],
                                             overall_results['semantic_set_ids'][:, :, :int(i)]))
    number_of_semantic_sets_on_subsets.append(
        get_number_of_unique_elements_per_row(overall_results['semantic_set_ids'][0][:, :i]))

average_pointwise_mutual_information = get_mean_of_poinwise_mutual_information(
    overall_results['pointwise_mutual_information'])

overall_results['mutual_information'] = mutual_information
overall_results['predictive_entropy'] = predictive_entropy
overall_results['predictive_entropy_over_concepts'] = predictive_entropy_over_concepts
overall_results['unnormalised_entropy_over_concepts'] = unnormalised_entropy_over_concepts
overall_results['number_of_semantic_sets'] = number_of_semantic_sets
overall_results['margin_measures'] = margin_measures
overall_results['unnormalised_margin_measures'] = unnormalised_margin_measures

overall_results['average_predictive_entropy'] = average_predictive_entropy
for i in range(len(average_predictive_entropy_on_subsets)):
    overall_results[f'average_predictive_entropy_on_subset_{i + 1}'] = average_predictive_entropy_on_subsets[i]
    overall_results[f'predictive_entropy_on_subset_{i + 1}'] = predictive_entropy_on_subsets[i]
    overall_results[f'semantic_predictive_entropy_on_subset_{i + 1}'] = semantic_predictive_entropy_on_subsets[i]
    overall_results[f'number_of_semantic_sets_on_subset_{i + 1}'] = number_of_semantic_sets_on_subsets[i]
overall_results['average_pointwise_mutual_information'] = average_pointwise_mutual_information

with open(f'{config.output_dir}sequences/{run_name}/aggregated_likelihoods_{args.generation_model}_generations.pkl',
          'wb') as outfile:
    pickle.dump(overall_results, outfile)

if args.verbose:
    print('Margin measure', margin_measures)
    print('Number of semantic sets', number_of_semantic_sets)
    print('predicitve entropy shape: ', predictive_entropy.shape)
    print('predicitve entropy per concept shape: ', predictive_entropy_over_concepts.shape)
    print(overall_results['average_neg_log_likelihoods'].shape)
    print(len(number_of_semantic_sets_on_subsets))
    print(number_of_semantic_sets_on_subsets[0].shape)
    print('average predictive entropy on subsets: ', len(average_predictive_entropy_on_subsets))
    print(average_predictive_entropy_on_subsets[0].shape)
    print(overall_results['pointwise_mutual_information'])
    print(overall_results['margin_measures'])

