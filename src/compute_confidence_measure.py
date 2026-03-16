import argparse
from ast import Raise
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
from scipy.special import softmax
from tqdm import tqdm 
from joblib import Parallel, delayed
from config import ANALYSIS_TEMP

MINIMIZE_EPS = 1e-3


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

path_prefix = f'{config.output_dir}sequences/{run_name}/'

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
    semantic_set_ids = semantic_set_ids.to(log_likelihoods.device)

    print("\n\nShape of log_likelihoods:",log_likelihoods.shape,"\n\n")
    # log_likelihoods is of size (1, Questions, Number of answers)
    # the next line just squeezes the tensor by removing the first dimension  
    # leaving us with a tensor of size (Questions, Number of answers)
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
        # compute entropy 
        # aggregated_likelihoods.shape[0] should be the cluster size (the numbmer of sequences in the cluster)
        entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0]) 
        entropies.append(entropy) # append entropy (entropy over the answers for one question)
        print("aggregated likelihoods:", aggregated_likelihoods, "Entropy:", entropy)
        

    return torch.tensor(entropies)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>                                                                    <<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>                         Credal Entropy                             <<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>                                                                    <<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def entropy(p, base: float=2):
    """Shannon Entropy in nats (default) or base."""
    # Clip to avoid log(0)
    p = np.clip(p, 1e-12, 1.0)
    if base == 2:
        return -np.sum(p * np.log2(p))
    return -np.sum(p * np.log(p))


def upper_entropy(probs_list: list, base: float = 2, n_jobs: int = -1) -> np.ndarray:
    """Compute the upper entropy of a credal set (Interval Method)."""

    def compute_upper_entropy(i: int) -> float:
        # Get the matrix for this specific question
        probs_matrix = probs_list[i]

        if probs_matrix.shape[1] < 2:
            return 0.0

        # Determine bounds from samples
        # lower_bound[k] = min probability observed for cluster k
        # upper_bound[k] = max probability observed for cluster k
        lower_bounds = np.min(probs_matrix, axis=0)
        upper_bounds = np.max(probs_matrix, axis=0)

        # Initial guess
        x0 = probs_matrix.mean(axis=0)

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        def fun(x: np.ndarray) -> float:
            return -entropy(x, base=base)

        bounds = list(zip(lower_bounds, upper_bounds))

        res = scipy.optimize.minimize(fun=fun, x0=x0, bounds=bounds, constraints=constraints)
        return float(-res.fun)

    if n_jobs:
        ue = Parallel(n_jobs=n_jobs)(
            delayed(compute_upper_entropy)(i) for i in tqdm(range(len(probs_list)), desc='Upper Entropy')
        )
        ue = np.array(ue)

    else:
        ue = np.empty(len(probs_list))
        for i in tqdm(range(len(probs_list)), desc='Upper Entropy'):
            ue[i] = compute_upper_entropy(i)

    return ue


def lower_entropy(probs_list: list, base: float=2, n_jobs: int=-1) -> np.ndarray:
    """Compute the lower entropy of a credal set (Interval Method)."""

    def compute_lower_entropy(i: int) -> float:
        # Get the matrix for this specific question (Samples x Clusters)
        probs_matrix = probs_list[i]

        if probs_matrix.shape[1] < 2:
            return 0.0

        lower_bounds = np.min(probs_matrix, axis=0)
        upper_bounds = np.max(probs_matrix, axis=0)

        # Initial guess
        x0 = probs_matrix.mean(axis=0)
        n_classes = x0.shape[0]

        # If the initial solution is uniform, slightly preturb it, becuase minimize will fail otherwise
        # (Minimizing entropy is concave, solvers get stuck on perfectly flat saddles)
        if np.all(np.isclose(x0, 1 / n_classes)):
            x0[0] += MINIMIZE_EPS
            x0[1] -= MINIMIZE_EPS

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        # Objective: Minimize Entropy
        def fun(x: np.ndarray) -> float:
            return entropy(x, base=base)

        bounds = list(zip(lower_bounds, upper_bounds))

        res = scipy.optimize.minimize(fun=fun, x0=x0, bounds=bounds, constraints=constraints)
        return float(res.fun)

    if n_jobs:
        le = Parallel(n_jobs=n_jobs)(
                delayed(compute_lower_entropy)(i) for i in tqdm(range(len(probs_list)), desc='Lower Entropy')
        )
        le = np.array(le)
    else:
        le = np.empty(len(probs_list))
        for i in tqdm(range(len(probs_list)), desc='Lower Entropy'):
            le[i] = compute_lower_entropy(i)

    return le



def get_credal_data_matrices(all_temperatures_likelihoods): 
    """
    Transforms the temperature dictionary into a list of Credal Intervals per question.
    """

    # 1. Pivot Data: Group by Question_ID
    # Structure: { q_id: { temp: sample_dict } }
    grouped_data = {}

    for temp, samples in all_temperatures_likelihoods.items():
        for sample in samples: 
            # Handle ID being list or int
            q_id = sample['id'][0] if isinstance(sample['id'], list) else sample['id']

            if q_id not in grouped_data:
                grouped_data[q_id] = {}

            grouped_data[q_id][temp] = sample

    # 2. Compute Probabilities & Bounds per Question
    probability_matrices_list = [] # List numpy arrays (K, 2)
    q_ids_list = []

    logging.info(f"Processing {len(grouped_data)} unique questions to compute bounds...")

    for q_id, temp_dict in grouped_data.items():
        # Find the total number of Semantic Sets (Clusters) for this question
        # We look at 'semantic_set_ids' across all temperatures to find the Max ID
        all_ids = []
        for sample in temp_dict.values():
        # semantic_set_ids is a tensor or list
            ids = sample['semantic_set_ids']
            if isinstance(ids, torch.Tensor):
                ids = ids.cpu().numpy()
            all_ids.extend(ids)

        if not all_ids:
            raise Exception("all_ids empty")

        num_clusters = max(all_ids) + 1
        num_temps=  len(temp_dict)

        # Matrix: Rows=Temps, Cols=Clusters
        # Stores P(Cluster_k | Temp_t)
        cluster_probs_matrix = np.zeros((num_temps, num_clusters))

        # Compute P(Cluster) for each temperature
        for i, (temp, sample) in enumerate(temp_dict.items()):
            # 1. Get Log Likelihoods (P(generation | context))
            # We use 'average_neg_log_likelihoods' -> NegLogLikelihood per token
            # To get Total Log Prob: -1 * avg_nll * length (or just use neg_log_likelihoods if available)
            
            # WARNING:
            # Using 'neg_log_likelihoods' (Total NLL) is mathematically safer for P(seq)
            # Input is POSITIVE NLL. We need NEGATIVE for log-prob.
            # I could also try using the average-neg-log-likelihoods
            nll = sample['neg_log_likelihoods']
            if isinstance(nll, torch.Tensor): nll = nll.cpu().numpy()

            # 3. Sum probabilities by Cluster ID
            ids = sample['semantic_set_ids']
            if isinstance(ids, torch.Tensor): ids = ids.cpu().numpy()

            # Sanity Check for NaNs
            logging.warning("NaN in:", nll)
            if np.isnan(nll).any() or np.isinf(nll).any():
                 nll = np.nan_to_num(nll, nan=1e9, posinf=1e9, neginf=1e9)

            # Convert to log-likelihood (negative value)
            # log_prob_seq = log( P(seq | context) )
            log_probs_seq = -1.0 * nll

            # 2. Compute Unnormalized Log-Probability for each Cluster
            # We use LogSumExp to sum probabilities in log-space:
            # log( P(Cluster_k) ) = log( sum_{seq in k} exp(log_prob_seq) )
            cluster_log_probs = np.full(num_clusters, -np.inf) # initialize with log(0)

            unique_cluster_ids = np.unique(ids)
            for cluster_id in unique_cluster_ids:
                # Get log_probs for all sequences belonging to this cluster
                seq_indices = np.where(ids == cluster_id)[0]
                cluster_seq_log_probs = log_probs_seq[seq_indices]

                # Sum them up (LogSumExp)
                # max_val trick for numerical stability: log(sum(exp(x))) = m + log(sum(exp(x-m)))
                max_val = np.max(cluster_seq_log_probs)
                if max_val == -np.inf:
                    cluster_log_mass = -np.inf
                else:
                    cluster_log_mass = max_val + np.log(np.sum(np.exp(cluster_seq_log_probs - max_val)))

                cluster_log_probs[cluster_id] = cluster_log_mass

            # 3. Normlaize across all clusters to get P(Cluster | Temp)
            # P(C_k) = exp( log(P(C_k)) ) / sum_j( exp( log(P(C_j)) ) )

            # Filter out -inf (empty clusters) for normalization calculation
            if np.all(cluster_log_probs == -np.inf):
                normalized_probs = np.zeros(num_clusters) # NOTE: Should not happen!!! (Sanity check)
            else:
                normalized_probs = softmax(cluster_log_probs)

            # Store in matrix
            cluster_probs_matrix[i, :] = normalized_probs

            # THIS BELONGS TO SOFTMAX
            #for seq_idx, cluster_id in enumerate(ids):
            #    cluster_probs_matrix[i, cluster_id] += norm_probs_seq[seq_idx]

        probability_matrices_list.append(cluster_probs_matrix)
        q_ids_list.append(q_id)



    print("\n\n\nprobability_matrices_list:\n", probability_matrices_list)
    return q_ids_list, probability_matrices_list


    """
    END
    """
    """
    semantic_set_ids = semantic_set_ids.to(log_likelihoods.device)
    print("\n\nShape of log_likelihoods:",log_likelihoods.shape,"\n\n")
    # log_likelihoods is of size (1, Questions, Number of answers)
    # the next line just squeezes the tensor by removing the first dimension  
    # leaving us with a tensor of size (Questions, Number of answers)
    M = log_likelihoods.shape[0]
    logM = torch.log(torch.tensor(M, dtype=log_likelihoods.dtype, device=log_likelihoods.device))

    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - logM

    semantic_set_ids = semantic_set_ids[0]

    for row_index in range(mean_across_models.shape[0]): # for each question do:
        row_log_probs = mean_across_models[row_index]
        semantic_set_ids_row = semantic_set_ids[row_index]

        aggregated_cluster_likelihoods = []
        bounds = []

        # NOTE: Should I normalize the sequence probabilities to compute the cluster lls and lower and upper bounds???
        # normalized_sequence_probs = torch.nn.functional.softmax(row_log_probs, dim=0)

        for semantic_set_id in torch.unique(semantic_set_ids_row): # for each cluster do:
            # compute cluster log-likelihood  
            aggregated_cluster_likelihood = torch.logsumexp(row_log_probs[semantic_set_ids_row == semantic_set_id], dim=0) # compute cluster likelihood
            aggregated_cluster_likelihoods.append(aggregated_cluster_likelihood) # store cluster likelihood





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
    """
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>                                                                    <<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def get_margin_probability_uncertainty_measure(log_likelihoods):
    """Compute margin probability uncertainty measure"""
    mean_across_models = torch.logsumexp(log_likelihoods, dim=0) - torch.log(torch.tensor(log_likelihoods.shape[0]))
    topk_likelihoods, indices = torch.topk(mean_across_models, 2, dim=1, sorted=True)
    margin_probabilities = np.exp(topk_likelihoods[:, 0]) - np.exp(topk_likelihoods[:, 1])

    return margin_probabilities


def get_number_of_unique_elements_per_row(tensor):
    assert len(tensor.shape) == 2
    return torch.count_nonzero(torch.sum(torch.nn.functional.one_hot(tensor), dim=1), dim=1)

"""
MAIN EXECUTION
"""

input_path = f'{path_prefix}{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl'
logging.info(f"Loading data from {input_path}")

with open(input_path, 'rb') as infile:
    all_temperatures_likelihoods = pickle.load(infile)

# RUN CREDAL LOGIC (Uses ALL temperatures)
# 1. Prepare Data
credal_ids, prob_matrices = get_credal_data_matrices(all_temperatures_likelihoods)

# 2. Run New Entropy Functions (Parallelized via joblib inside function)
# Note: Using base=e (nats) to match standard Pytorch entropy, 
# or base=2 (bits) if you prefer. Your new functions default to 2.
lower_entropies = lower_entropy(prob_matrices, base=np.e) 
upper_entropies = upper_entropy(prob_matrices, base=np.e)

epistemic_uncertainty = upper_entropies - lower_entropies

# Map credal results to ID for easy lookup
# Map credal results to ID for easy lookup
credal_results_map = {
    id_: {
        'lower_entropy': lower_entropies[i],
        'upper_entropy': upper_entropies[i],
        'epistemic_uncertainty': epistemic_uncertainty[i]
    }
    for i, id_ in enumerate(credal_ids)
}

list_of_results = []

# ---------------------------------------------------------
# COMPREHENSIVE BASELINE EVALUATION
# ---------------------------------------------------------
# We want to see how baselines perform at ALL temps, not just one.

baseline_results_per_temp = {}

# Get list of all available temperatures from your data
# Assuming all_temperatures_likelihoods keys are temperatures (float or str)
available_temps = list(all_temperatures_likelihoods.keys())

for temp in available_temps:
    logging.info(f"Processing Baselines for Temperature: {temp}")
    
    # Extract the samples for this specific temperature
    # Note: You need to ensure the order of IDs matches your overall results
    # It is safer to re-construct the list_of_results logic for each temp
    
    current_temp_samples = all_temperatures_likelihoods[temp]
    
    # Sort or align by ID if necessary, assuming they are aligned here
    list_of_results_temp = [(args.evaluation_model, current_temp_samples)]
    
    # Get standard metrics
    overall_res_temp = get_overall_log_likelihoods(list_of_results_temp)
    
    # Compute Semantic Entropy (Predictive Entropy over Concepts)
    # This is your main baseline comparison
    sem_entropy = get_predictive_entropy_over_concepts(
        -overall_res_temp['average_neg_log_likelihoods'],
        overall_res_temp['semantic_set_ids']
    )
    
    # Compute Standard Predictive Entropy
    pred_entropy = get_predictive_entropy(
        -overall_res_temp['neg_log_likelihoods']
    )
    margin_measures = get_margin_probability_uncertainty_measure(
        -overall_res_temp['average_neg_log_likelihoods']
    )
    unnormalised_margin_measures = get_margin_probability_uncertainty_measure(
        -overall_res_temp['neg_log_likelihoods']
    )
    mutual_information = get_mutual_information(
        -overall_res_temp['neg_log_likelihoods']
    )

    # Store specifically mapped by ID
    temp_res_map = {}
    for i, _id in enumerate(overall_res_temp['ids']):
        # Handle ID list vs int issue
        val_id = _id[0] if isinstance(_id, list) else _id
        
        temp_res_map[val_id] = {
            'semantic_entropy': sem_entropy[i],
            'predictive_entropy': pred_entropy[i],
            'margin_measures': margin_measures[i],
            'unnormalised_margin_measures': unnormalised_margin_measures[i],
            'mutual_information': mutual_information[i],
        }
    
    baseline_results_per_temp[temp] = temp_res_map

# ---------------------------------------------------------
# MERGE INTO OVERALL RESULTS
# ---------------------------------------------------------

# WARNING: RUN STANDARD LOGIC (Uses a single representative temperature)
with open(f'{path_prefix}{args.generation_model}_ANALYSIS_TEMP_generations_{args.evaluation_model}_likelihoods.pkl',
          'rb') as infile:
    sequences = pickle.load(infile)
    list_of_results.append((args.evaluation_model, sequences))
# Format data for the old 'get_overall_log_likelihoods' function
overall_results = get_overall_log_likelihoods(list_of_results)

"""
END
"""

#with open(input_path,'rb') as infile:
#    sequences = pickle.load(infile)
#    list_of_results.append((args.evaluation_model, sequences))

#overall_results = get_overall_log_likelihoods(list_of_results)
mutual_information = get_mutual_information(-overall_results['neg_log_likelihoods'])
predictive_entropy = get_predictive_entropy(-overall_results['neg_log_likelihoods'])
predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['average_neg_log_likelihoods'],
                                                                        overall_results['semantic_set_ids'])
unnormalised_entropy_over_concepts = get_predictive_entropy_over_concepts(-overall_results['neg_log_likelihoods'],
                                                                          overall_results['semantic_set_ids'])
margin_measures = get_margin_probability_uncertainty_measure(-overall_results['average_neg_log_likelihoods'])
unnormalised_margin_measures = get_margin_probability_uncertainty_measure(-overall_results['neg_log_likelihoods'])

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

"""
MERGE RESULTS
"""
# We align the Credal results with the Standard results using IDs
# (Assuming the order of IDs in overall_results matches the input list, but we double check)
num_samples = len(overall_results['ids'])
credal_eu_tensor = torch.zeros(num_samples)
lower_ent_tensor = torch.zeros(num_samples)
upper_ent_tensor = torch.zeros(num_samples)

for idx, id_ in enumerate(overall_results['ids']):
    if id_ in credal_results_map:
        res = credal_results_map[id_]
        credal_eu_tensor[idx] = res['epistemic_uncertainty']
        lower_ent_tensor[idx] = res['lower_entropy']
        upper_ent_tensor[idx] = res['upper_entropy']

# NOTE: Add to the dictionary
overall_results['credal_epistemic_uncertainty'] = credal_eu_tensor
overall_results['credal_lower_entropy'] = lower_ent_tensor
overall_results['credal_upper_entropy'] = upper_ent_tensor

overall_results['baselines_all_temps'] = baseline_results_per_temp # Save all for plotting later
"""
END
"""

with open(f'{path_prefix}aggregated_likelihoods_{args.generation_model}_generations.pkl',
          'wb') as outfile:
    pickle.dump(overall_results, outfile)

if args.verbose:
    print("\n--- Summary ---")
    print(f"Standard Metrics (T={ANALYSIS_TEMP}) computed for {num_samples} samples.")
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
    print(f"Credal Metrics (All Temps) computed and merged.")
    print(f"Mean Credal Epistemic Uncertainty: {torch.mean(credal_eu_tensor):.4f}")
