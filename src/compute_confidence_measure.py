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
from tqdm import tqdm 
from joblib import Parallel, delayed
from config import ANALYSIS_TEMP

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
def entropy(p):
    """
    Helper: Shannon Entropy in nats (base e). Handles 0 log 0
    Add a tiny epsilon to avoid log(0), or use specialized func
    """
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

    # sum(lower) must be <= 1 and sum(upper) must be >= 1
    # VALID Condition: sum(lower) <= 1 AND sum(upper) >= 1
    # We add epsilon (1e-6) to handle floating point noise.
    # assert that sum of lower bounds is NOT significantly greater than 1
    assert np.sum(lower_bounds) <= 1.0 + 1e-6, f"Invalid Set: Sum of lower bounds is {np.sum(lower_bounds)}"
    # assert that sum of upper bounds is NOT significantly less than 1
    assert np.sum(upper_bounds) >= 1.0 - 1e-6, f"Invalid Set: Sum of upper bounds is {np.sum(upper_bounds)}"

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



def batched_entropy_diff(list_of_bounds, batch_size=128, n_jobs=-1):
    """
    Computes (Upper Entropy - Lower Entropy) for a batch of Credal intervals.
    Expects intervals of shape (N_samples, N_classes, 2).
    """
    # Use Joblib to parallelize the expensive scipy optimization
    # This replaces the batch loop with parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(solve_credal_entropy)(bounds)
        for bounds in tqdm(list_of_bounds, desc='Optimizing Credal Entropy')
    )

    results = np.array(results) # Shape (N, 2)

    # results[:, 0] is Lower Entropy
    # results[:, 1] is Upper Entropy
    diffs = results[:, 1] - results[:, 0]

    return diffs


def get_credal_entropy_over_concepts(all_temperatures_likelihoods): 
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
    credal_bounds_list = [] # List numpy arrays (K, 2)
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
            
            # Using 'neg_log_likelihoods' (Total NLL) is mathematically safer for P(seq)
            # Input is POSITIVE NLL. We need NEGATIVE for log-prob.
            nll = sample['neg_log_likelihoods']
            if isinstance(nll, torch.Tensor): nll = nll.cpu().numpy()

            log_probs_seq = -1.0 * nll

            # 2. Normalize sequences to get P(seq | temp)
            # Softmax: exp(x) / sum(exp(x))
            max_log = np.max(log_probs_seq)
            exp_probs = np.exp(log_probs_seq - max_log)
            sum_exp = np.sum(exp_probs)
            norm_probs_seq = exp_probs / sum_exp

            # 3. Sum probabilities by Cluster ID
            ids = sample['semantic_set_ids']
            if isinstance(ids, torch.Tensor): ids = ids.cpu().numpy()

            for seq_idx, cluster_id in enumerate(ids):
                cluster_probs_matrix[i, cluster_id] += norm_probs_seq[seq_idx]

        # Compute Bounds
        # Lower Bound: min prob across rows
        # Upper bound: max prob across rows
        lower_bounds = np.min(cluster_probs_matrix, axis=0)
        upper_bounds = np.max(cluster_probs_matrix, axis=0)

        # Stack: (K, 2)
        bounds = np.stack([lower_bounds, upper_bounds], axis=1)
        credal_bounds_list.append(bounds)
        q_ids_list.append(q_id)

    # Solve Optimization (Parallel)
    logging.info("Optimizing Credal Entropy...")
    results = Parallel(n_jobs=-1)(delayed(solve_credal_entropy)(b) for b in tqdm(credal_bounds_list))
    results = np.array(results) # [LowerEnt, UpperEnt]
    return q_ids_list, results, credal_bounds_list


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

input_path = f'{config.output_dir}sequences/{run_name}/{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl'
logging.info(f"Loading data from {input_path}")

with open(input_path, 'rb') as infile:
    all_temperatures_likelihoods = pickle.load(infile)

# RUN CREDAL LOGIC (Uses ALL temperatures)
credal_ids, credal_entropy_results, credal_bounds = get_credal_entropy_over_concepts(all_temperatures_likelihoods)
epistemic_uncertainty = credal_entropy_results[:, 1] - credal_entropy_results[:, 0]

# Map credal results to ID for easy lookup
credal_results_map = {
    id_: {
        'lower_entropy': credal_entropy_results[i, 0],
        'upper_entropy': credal_entropy_results[i, 1],
        'epistemic_uncertainty': epistemic_uncertainty[i]
    }
    for i, id_ in enumerate(credal_ids)
}

list_of_results = []

# WARNING: RUN STANDARD LOGIC (Uses a single representative temperature)
with open(f'{config.output_dir}sequences/{run_name}/{args.generation_model}_ANALYSIS_TEMP_generations_{args.evaluation_model}_likelihoods.pkl',
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
"""
END
"""

with open(f'{config.output_dir}sequences/{run_name}/aggregated_likelihoods_{args.generation_model}_generations.pkl',
          'wb') as outfile:
    pickle.dump(overall_results, outfile)

if args.verbose:
    print("\n--- Summary ---")
    print(f"Standard Metrics (T={target_temp}) computed for {num_samples} samples.")
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
