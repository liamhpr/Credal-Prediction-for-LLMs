# parse arguments
import argparse
import json
import pickle
import os

import config
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import wandb
from config import ANALYSIS_TEMP

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--run_ids', nargs='+', default=[])
parser.add_argument('--verbose', type=bool, default=True)
args = parser.parse_args()

overall_result_dict = {}

aurocs_across_models = []

sequence_embeddings_dict = {}

run_ids_to_analyze = args.run_ids
for run_id in run_ids_to_analyze:

    wandb.init(project='credal-prediction-for-large-language-models', id=run_id, resume='allow')
    run_name = wandb.run.name
    model_name = wandb.config.model
    print(f"Analyzing Run: {run_name} | Model: {model_name}")
    path_prefix = f'{config.output_dir}sequences/{run_name}/'


    def get_samples_for_temp(pkl_path, temp):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and isinstance(list(data.keys())[0], float):
            if temp in data:
                return data[temp]
            else: 
                # Fallback :Use the first available key
                fallback = list(data.keys())[0]
                print(f"Warning: Temp {temp} not found in {pkl_path}. Using {fallback}.")
                return data[fallback]
        return data

    def get_similarities_df_credal():
        opt_temp_path = f'{path_prefix}{model_name}_optimal_temperature.pkl'
        if os.path.exists(opt_temp_path):
            with open(opt_temp_path, 'rb') as f:
                best_temp = pickle.load(f)
        else:
            print("Optimal temperature file not found. Cannot load similarities.")
            return pd.DataFrame()

        sim_path = f'{path_prefix}{model_name}_T{best_temp}_generations_similarities.pkl'

        if not os.path.exists(sim_path):
            print(f"Similarity file not found: {sim_path}")
            return pd.DataFrame()

        with open(sim_path, 'rb') as f:
            similarities = pickle.load(f)

        similarities_df = pd.DataFrame.from_dict(similarities, orient='index')
        similarities_df['id'] = similarities_df.index

        similarities_df['id'] = similarities_df['id'].apply(lambda x: x[0] if isinstance(x, list) else x)
        similarities_df['id'] = similarities_df['id'].astype('object')

        if 'has_semantically_different_answers' in similarities_df.columns:
            similarities_df['has_semantically_different_answers'] = similarities_df['has_semantically_different_answers'].astype('int')

        if 'syntactic_similarities' in similarities_df.columns:
            similarities_df['rougeL_among_generations_credal'] = similarities_df['syntactic_similarities'].apply(lambda x: x['rougeL'])

        return similarities_df


    def get_similarities_df():
        """Get the similarities df from the pickle file"""
        with open(f'{path_prefix}{model_name}_ANALYSIS_TEMP_generations_similarities.pkl', 'rb') as f:
            similarities = pickle.load(f)
            similarities_df = pd.DataFrame.from_dict(similarities, orient='index')
            similarities_df['id'] = similarities_df.index
            similarities_df['has_semantically_different_answers'] = similarities_df[
                'has_semantically_different_answers'].astype('int')
            similarities_df['rougeL_among_generations'] = similarities_df['syntactic_similarities'].apply(
                lambda x: x['rougeL'])

            return similarities_df


    def get_generations_df():
        """Get the generations df from the pickle file"""
        #path = f'{config.output_dir}/{run_name}/{model_name}_generations.pkl'
        #samples = get_samples_for_temp(path, ANALYSIS_TEMP)

        #generations_df = pd.DataFrame(samples)
        with open(f'{path_prefix}{model_name}_ANALYSIS_TEMP_generations.pkl', 'rb') as infile:
            generations = pickle.load(infile)
            generations_df = pd.DataFrame(generations)

            # --- ROBUST EXTRACTION FUNCTION ---
            def safe_extract(x):
                # Layer 1: Peel outer list
                if isinstance(x, (list, tuple, np.ndarray)):
                    if len(x) == 0: return None
                    x = x[0]
                
                # Layer 2: Peel inner list (Fixes your specific error)
                if isinstance(x, (list, tuple, np.ndarray)):
                    if len(x) == 0: return None
                    x = x[0]

                # Layer 3: Extract value from Tensor
                if hasattr(x, 'item'):
                    return x.item()

                return x
            # ----------------------------------

            generations_df['id'] = generations_df['id'].apply(lambda x: x[0])
            generations_df['id'] = generations_df['id'].apply(lambda x: x[0] if isinstance(x, list) else x)
            generations_df['id'] = generations_df['id'].astype('object')
            if not generations_df['semantic_variability_reference_answers'].isnull().values.any():
                generations_df['semantic_variability_reference_answers'] = generations_df[
                    'semantic_variability_reference_answers'].apply(safe_extract)

            if not generations_df['rougeL_reference_answers'].isnull().values.any():
                generations_df['rougeL_reference_answers'] = generations_df['rougeL_reference_answers'].apply(safe_extract)

            generations_df['length_of_most_likely_generation'] = generations_df['most_likely_generation'].apply(
                lambda x: len(str(x).split(' ')))
            generations_df['length_of_answer'] = generations_df['answer'].apply(lambda x: len(str(x).split(' ')))
            generations_df['variance_of_length_of_generations'] = generations_df['generated_texts'].apply(
                lambda x: np.var([len(str(y).split(' ')) for y in x]))
            generations_df['correct'] = (generations_df['rougeL_to_target'] > 0.3).astype('int')

            return generations_df


    def get_generations_df_credal():
        """Get the generations df for the best temperature from the dictionary pickle"""
        
        # 1. Load the dictionary containing ALL temperatures
        # Note: Ensure this matches the filename saved in generate.py
        dict_path = f'{path_prefix}{model_name}_all_generations.pkl'
        with open(dict_path, 'rb') as infile:
            all_generations_dict = pickle.load(infile)

        # 2. Determine the Best Temperature
        # Option A: If you saved it to a file in generate.py
        opt_temp_path = f'{path_prefix}{model_name}_optimal_temperature.pkl'
        if os.path.exists(opt_temp_path):
            with open(opt_temp_path, 'rb') as f:
                best_temp = pickle.load(f)
                print(f"Loaded optimal temperature: {best_temp}")
        else:
            # Option B: Hardcode it or pick a default (e.g., 0.5 or 0.1)
            # If the dict keys are floats, pick the one you want.
            # For now, let's fallback to the first key if file missing
            best_temp = list(all_generations_dict.keys())[0]
            print(f"Optimal temp file not found. Using fallback: {best_temp}")

        # 3. Extract the specific samples for that temperature
        if best_temp in all_generations_dict:
            samples = all_generations_dict[best_temp]
        else:
            raise ValueError(f"Temperature {best_temp} not found in {dict_path}")

        generations_df = pd.DataFrame(samples)

        # --- ROBUST EXTRACTION FUNCTION ---
        def safe_extract(x):
            # Layer 1: Peel outer list
            if isinstance(x, (list, tuple, np.ndarray)):
                if len(x) == 0: return None
                x = x[0]
            
            # Layer 2: Peel inner list
            if isinstance(x, (list, tuple, np.ndarray)):
                if len(x) == 0: return None
                x = x[0]

            # Layer 3: Extract value from Tensor
            if hasattr(x, 'item'):
                return x.item()
            
            return x
        # ----------------------------------

        generations_df['id'] = generations_df['id'].apply(lambda x: x[0])
        generations_df['id'] = generations_df['id'].apply(lambda x: x[0] if isinstance(x, list) else x)
        generations_df['id'] = generations_df['id'].astype('object')
        if not generations_df['semantic_variability_reference_answers'].isnull().values.any():
            generations_df['semantic_variability_reference_answers'] = generations_df[
                'semantic_variability_reference_answers'].apply(safe_extract)

        if not generations_df['rougeL_reference_answers'].isnull().values.any():
            generations_df['rougeL_reference_answers'] = generations_df['rougeL_reference_answers'].apply(safe_extract)
        generations_df['length_of_most_likely_generation'] = generations_df['most_likely_generation'].apply(
            lambda x: len(str(x).split(' ')))
        generations_df['length_of_answer'] = generations_df['answer'].apply(lambda x: len(str(x).split(' ')))
        generations_df['variance_of_length_of_generations'] = generations_df['generated_texts'].apply(
            lambda x: np.var([len(str(y).split(' ')) for y in x]))
        generations_df['correct'] = (generations_df['rougeL_to_target'] > 0.3).astype('int')

        return generations_df

    def get_generations_df_specific_temperature(temperature):
        """Get the generations df for the best temperature from the dictionary pickle"""
        
        # 1. Load the dictionary containing ALL temperatures
        # Note: Ensure this matches the filename saved in generate.py
        dict_path = f'{path_prefix}{model_name}_all_generations.pkl'
        with open(dict_path, 'rb') as infile:
            all_generations_dict = pickle.load(infile)

        # 2. Extract the specific samples for that temperature
        if temperature in all_generations_dict:
            samples = all_generations_dict[temperature]
        else:
            raise ValueError(f"Temperature {temperature} not found in {dict_path}")

        generations_df = pd.DataFrame(samples)

        # --- ROBUST EXTRACTION FUNCTION ---
        def safe_extract(x):
            # Layer 1: Peel outer list
            if isinstance(x, (list, tuple, np.ndarray)):
                if len(x) == 0: return None
                x = x[0]
            
            # Layer 2: Peel inner list
            if isinstance(x, (list, tuple, np.ndarray)):
                if len(x) == 0: return None
                x = x[0]

            # Layer 3: Extract value from Tensor
            if hasattr(x, 'item'):
                return x.item()
            
            return x
        # ----------------------------------

        generations_df['id'] = generations_df['id'].apply(lambda x: x[0])
        generations_df['id'] = generations_df['id'].apply(lambda x: x[0] if isinstance(x, list) else x)
        generations_df['id'] = generations_df['id'].astype('object')
        if not generations_df['semantic_variability_reference_answers'].isnull().values.any():
            generations_df['semantic_variability_reference_answers'] = generations_df[
                'semantic_variability_reference_answers'].apply(safe_extract)

        if not generations_df['rougeL_reference_answers'].isnull().values.any():
            generations_df['rougeL_reference_answers'] = generations_df['rougeL_reference_answers'].apply(safe_extract)
        generations_df['length_of_most_likely_generation'] = generations_df['most_likely_generation'].apply(
            lambda x: len(str(x).split(' ')))
        generations_df['length_of_answer'] = generations_df['answer'].apply(lambda x: len(str(x).split(' ')))
        generations_df['variance_of_length_of_generations'] = generations_df['generated_texts'].apply(
            lambda x: np.var([len(str(y).split(' ')) for y in x]))
        generations_df['correct'] = (generations_df['rougeL_to_target'] > 0.3).astype('int')

        return generations_df

    def get_likelihoods_df():
        """Get the likelihoods df from the pickle file"""

        with open(f'{path_prefix}aggregated_likelihoods_{model_name}_generations.pkl', 'rb') as f:
            likelihoods = pickle.load(f)
            print("Loaded likelihood keys:", likelihoods.keys())

            subset_keys = ['average_predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
            subset_keys += ['predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
            subset_keys += ['semantic_predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
            subset_keys += ['number_of_semantic_sets_on_subset_' + str(i) for i in range(1, num_generations + 1)]

            keys_to_use = ('ids', 'predictive_entropy', 'mutual_information', 'average_predictive_entropy',\
                            'average_pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                            #'average_neg_log_likelihood_of_second_most_likely_gen', 
                            'neg_log_likelihood_of_most_likely_gen',\
                            'predictive_entropy_over_concepts', 'number_of_semantic_sets', 'unnormalised_entropy_over_concepts',\
                            'credal_epistemic_uncertainty', 'credal_lower_entropy', 'credal_upper_entropy')

            available_keys = []
            for k in keys_to_use: 
                if k in likelihoods:
                    available_keys.append(k)
                else:
                    print("KEYERROR:", k, "not found in likelihoods")

            likelihoods_small = dict((k, likelihoods[k]) for k in available_keys + list(set(subset_keys) & set(likelihoods.keys())))
            for key in likelihoods_small:
                if key == 'average_predictive_entropy_on_subsets':
                    likelihoods_small[key].shape
                if type(likelihoods_small[key]) is torch.Tensor:
                    likelihoods_small[key] = torch.squeeze(likelihoods_small[key].cpu())

            sequence_embeddings = likelihoods['sequence_embeddings']

            likelihoods_df = pd.DataFrame.from_dict(likelihoods_small)

            likelihoods_df.rename(columns={'ids': 'id'}, inplace=True)

            if 'baselines_all_temps' in likelihoods:
                baseline_temps = likelihoods['baselines_all_temps']
                print(f"Found baselines for temperatures: {list(baseline_temps.keys())}")

                likelihoods_df['temp_map_id'] = likelihoods_df['id'].apply(lambda x: x[0] if isinstance(x, list) else x)

                for temp, id_map in baseline_temps.items():
                # Extract the first sample to see available metrics (e.g., semantic, margin, etc.)
                    first_key = list(id_map.keys())[0]
                    available_metrics = id_map[first_key].keys()

                    for metric in available_metrics:
                        # e.g., semantic_entropy_T0.5, margin_measures_T1.0
                        col_name = f"{metric}_T{temp}"

                        # Create mapping dict: {id: value}
                        # We handle tensors here immediately
                        val_map = {}
                        for _id, metrics_dict in id_map.items():
                            clean_id = _id[0] if isinstance(_id, list) else _id
                            val = metrics_dict[metric]
                            if hasattr(val, 'item'): val = val.item()
                            val_map[clean_id] = val

                        likelihoods_df[col_name] = likelihoods_df['temp_map_id'].map(val_map)

                del likelihoods_df['temp_map_id']

            return likelihoods_df, sequence_embeddings

    similarities_df = get_similarities_df()
    similarities_df_credal = get_similarities_df_credal()
    generations_df = get_generations_df()
    generations_df_credal = get_generations_df_credal()
    num_generations = len(generations_df['generated_texts'][0])
    likelihoods_df, sequence_embeddings = get_likelihoods_df()
    result_df = generations_df.merge(similarities_df, on='id').merge(likelihoods_df, on='id')
    #result_df_credal = generations_df_credal.merge(likelihoods_df, on='id')
    result_df_credal = generations_df_credal.merge(similarities_df_credal, on='id').merge(likelihoods_df, on='id')

    n_samples_before_filtering = len(result_df)
    result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))

    # =============== DATA CLEANING BLOCK ==================
    # 1. Identify columns needed for AUROC
    cols_to_check = [
        'correct',
        'average_predictive_entropy',
        'predictive_entropy',
        'predictive_entropy_over_concepts',
        'neg_log_likelihood_of_most_likely_gen',
        'number_of_semantic_sets',
        'rougeL_among_generations',
        'average_neg_log_likelihood_of_most_likely_gen'
    ]
    if 'unnormalised_entropy_over_concepts' in result_df.columns: 
        cols_to_check.append('unnormalised_entropy_over_concepts')

    opt_temp_path = f'{path_prefix}{model_name}_optimal_temperature.pkl'
    if os.path.exists(opt_temp_path):
        with open(opt_temp_path, 'rb') as f:
            best_temp = pickle.load(f)

    credal_cols_to_check = ['correct', 'credal_epistemic_uncertainty', f'predictive_entropy_T{best_temp}', f'semantic_entropy_T{best_temp}',
                            f'margin_measures_T{best_temp}', f'unnormalised_margin_measures_T{best_temp}', f'mutual_information_T{best_temp}', f'ln_predictive_entropy_T{best_temp}']


    valid_standard = result_df.dropna(subset=cols_to_check)
    valid_credal = result_df_credal.dropna(subset=credal_cols_to_check)

    common_vaid_ids = set(valid_standard['id']).intersection(set(valid_credal['id']))

    result_df = result_df[result_df['id'].isin(common_vaid_ids)].copy()
    result_df_credal = result_df_credal[result_df_credal['id'].isin(common_vaid_ids)].copy()

    if args.verbose:
        print(f'Removed {n_samples_before_filtering - len(result_df)} mismatched/NaN rows.')
        print(f'Remaining identical samples for comparison: {len(result_df)}')

    # ============ END OF DATA CLEANING BLOCK ==============

    # Begin analysis
    result_dict = {}
    result_dict['accuracy'] = result_df['correct'].mean()

    # WARNING: IMPLEMENT THE CREDAL ENTROPY OVER CONCEPTS 
    credal_entropy_over_concepts_auroc = sklearn.metrics.roc_auc_score(1 - result_df_credal['correct'],
                                                                   result_df_credal['credal_upper_entropy'])
    result_dict['credal_entropy_over_concepts_auroc'] = credal_entropy_over_concepts_auroc
    # WARNING:                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print("credal_entropy_over_concepts_auroc:", credal_entropy_over_concepts_auroc)

    credal_entropy_over_epistemic_uncertainty_auroc = sklearn.metrics.roc_auc_score(1 - result_df_credal['correct'],
                                                                    result_df_credal['credal_epistemic_uncertainty'])
    result_dict["credal_entropy_over_epistemic_uncertainty_auroc"] = credal_entropy_over_epistemic_uncertainty_auroc
    print("credal_entropy_over_epistemic_uncertainty_auroc:", credal_entropy_over_epistemic_uncertainty_auroc)

    credal_entropy_over_aleatoric_uncertainty_auroc = sklearn.metrics.roc_auc_score(1 - result_df_credal['correct'],
                                                                    result_df_credal['credal_lower_entropy'])
    result_dict['credal_entropy_over_aleatoric_uncertainty_auroc'] = credal_entropy_over_aleatoric_uncertainty_auroc
    print("credal_entropy_over_aleatoric_uncertainty_auroc:", credal_entropy_over_aleatoric_uncertainty_auroc)


    # Compute the auroc for the length normalized predictive entropy
    ln_predictive_entropy_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                result_df['average_predictive_entropy'])
    result_dict['ln_predictive_entropy_auroc'] = ln_predictive_entropy_auroc

    predictive_entropy_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'], result_df['predictive_entropy'])
    result_dict['predictive_entropy_auroc'] = predictive_entropy_auroc

    entropy_over_concepts_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                result_df['predictive_entropy_over_concepts'])
    result_dict['entropy_over_concepts_auroc'] = entropy_over_concepts_auroc


    if 'unnormalised_entropy_over_concepts' in result_df.columns:
        unnormalised_entropy_over_concepts_auroc = sklearn.metrics.roc_auc_score(
            1 - result_df['correct'], result_df['unnormalised_entropy_over_concepts'])
        result_dict['unnormalised_entropy_over_concepts_auroc'] = unnormalised_entropy_over_concepts_auroc

    aurocs_across_models.append(entropy_over_concepts_auroc)

    neg_llh_most_likely_gen_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                  result_df['neg_log_likelihood_of_most_likely_gen'])
    result_dict['neg_llh_most_likely_gen_auroc'] = neg_llh_most_likely_gen_auroc

    number_of_semantic_sets_auroc = sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                                                  result_df['number_of_semantic_sets'])
    result_dict['number_of_semantic_sets_auroc'] = number_of_semantic_sets_auroc

    result_dict['number_of_semantic_sets_correct'] = result_df[result_df['correct'] ==
                                                               1]['number_of_semantic_sets'].mean()
    result_dict['number_of_semantic_sets_incorrect'] = result_df[result_df['correct'] ==
                                                                 0]['number_of_semantic_sets'].mean()

    result_dict['average_rougeL_among_generations'] = result_df['rougeL_among_generations'].mean()
    result_dict['average_rougeL_among_generations_correct'] = result_df[result_df['correct'] ==
                                                                        1]['rougeL_among_generations'].mean()
    result_dict['average_rougeL_among_generations_incorrect'] = result_df[result_df['correct'] ==
                                                                          0]['rougeL_among_generations'].mean()
    result_dict['average_rougeL_auroc'] = sklearn.metrics.roc_auc_score(result_df['correct'],
                                                                        result_df['rougeL_among_generations'])

    average_neg_llh_most_likely_gen_auroc = sklearn.metrics.roc_auc_score(
        1 - result_df['correct'], result_df['average_neg_log_likelihood_of_most_likely_gen'])
    result_dict['average_neg_llh_most_likely_gen_auroc'] = average_neg_llh_most_likely_gen_auroc
    result_dict['rougeL_based_accuracy'] = result_df['correct'].mean()

    result_dict['margin_measure_auroc'] = sklearn.metrics.roc_auc_score(
        1 - result_df['correct'], result_df['average_neg_log_likelihood_of_most_likely_gen'])
        #+ result_df['average_neg_log_likelihood_of_second_most_likely_gen'])


    # WARNING: My logic to compute semantic entropy over most "optimal temperature"
    """
    opt_temp_path = f'{path_prefix}{model_name}_optimal_temperature.pkl'
    if os.path.exists(opt_temp_path):
        with open(opt_temp_path, 'rb') as f:
            best_temp = pickle.load(f)
            print(f"Loaded optimal temperature: {best_temp}")

            entropy_over_concepts_auroc_optimal_temperature = sklearn.metrics.roc_auc_score(1 - result_df_credal['correct'],
                                                                        result_df_credal[f'semantic_entropy_T{best_temp}'])
            result_dict[f'entropy_over_concepts_T{best_temp}_auroc'] = entropy_over_concepts_auroc_optimal_temperature

            predictive_entropy_auroc_optimal_temperature = sklearn.metrics.roc_auc_score(1 - result_df_credal['correct'], result_df_credal[f'predictive_entropy_T{best_temp}'])
            result_dict[f'predictive_entropy_T{best_temp}_auroc'] = predictive_entropy_auroc_optimal_temperature
    else:
        print("FAILED TO LOAD", opt_temp_path, "and compute semantic entropy and predictive entropy")
    """
    # -------------------------------------------------------------------------
    # ROBUST BLOCK: Compute AUROC for Optimal Temperature Baselines
    # -------------------------------------------------------------------------
    opt_temp_path = f'{path_prefix}{model_name}_optimal_temperature.pkl'

    if os.path.exists(opt_temp_path):
        with open(opt_temp_path, 'rb') as f:
            best_temp = pickle.load(f)
            print(f"Loaded optimal temperature: {best_temp}")

        # Helper function to extract float from Tensor/Object safely
        def safe_to_float(x):
            try:
                if hasattr(x, 'item'):
                    return x.item() # Handle PyTorch Tensors
                return float(x)
            except (ValueError, TypeError):
                return np.nan

        # ---------------------------------
        # 1. Compute Semantic Entropy AUROC
        # ---------------------------------
        col_name_sem = f'semantic_entropy_T{best_temp}'

        if col_name_sem in result_df_credal.columns:
            # Copy relevant columns
            temp_df = result_df_credal[['correct', col_name_sem]].copy()

            # STEP 1: Force conversion to pure Python floats (Removes Tensors)
            temp_df[col_name_sem] = temp_df[col_name_sem].apply(safe_to_float)

            # STEP 2: Force numeric type (Coerces any remaining junk to NaN)
            temp_df[col_name_sem] = pd.to_numeric(temp_df[col_name_sem], errors='coerce')

            # STEP 3: Handle Infinity and NaN
            # (Replace infinite values with NaN so dropna catches them)
            temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            temp_df.dropna(inplace=True)

            if len(temp_df) > 0:
                entropy_over_concepts_auroc_optimal_temperature = sklearn.metrics.roc_auc_score(
                    1 - temp_df['correct'],
                    temp_df[col_name_sem]
                )
                result_dict[f'entropy_over_concepts_T{best_temp}_auroc'] = entropy_over_concepts_auroc_optimal_temperature
                print(f"Success: {col_name_sem} AUROC = {entropy_over_concepts_auroc_optimal_temperature:.4f}")
            else:
                print(f"Warning: No valid samples left for {col_name_sem} after cleaning.")
        else:
            print(f"Warning: Column {col_name_sem} not found in DataFrame.")

        # ---------------------------------
        # 2. Compute Predictive Entropy AUROC
        # ---------------------------------
        col_name_pred = f'predictive_entropy_T{best_temp}'

        if col_name_pred in result_df_credal.columns:
            temp_df = result_df_credal[['correct', col_name_pred]].copy()

            # Same 3-step cleaning process
            temp_df[col_name_pred] = temp_df[col_name_pred].apply(safe_to_float)
            temp_df[col_name_pred] = pd.to_numeric(temp_df[col_name_pred], errors='coerce')
            temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            temp_df.dropna(inplace=True)

            if len(temp_df) > 0:
                predictive_entropy_auroc_optimal_temperature = sklearn.metrics.roc_auc_score(
                    1 - temp_df['correct'],
                    temp_df[col_name_pred]
                )
                result_dict[f'predictive_entropy_T{best_temp}_auroc'] = predictive_entropy_auroc_optimal_temperature
                print(f"Success: {col_name_pred} AUROC = {predictive_entropy_auroc_optimal_temperature:.4f}")
            else:
                print(f"Warning: No valid samples left for {col_name_pred} after cleaning.")
        else:
            print(f"Warning: Column {col_name_pred} not found in DataFrame.")

        col_name_pred = f'margin_measures_T{best_temp}'

        if col_name_pred in result_df_credal.columns:
            temp_df = result_df_credal[['correct', col_name_pred]].copy()

            # Same 3-step cleaning process
            temp_df[col_name_pred] = temp_df[col_name_pred].apply(safe_to_float)
            temp_df[col_name_pred] = pd.to_numeric(temp_df[col_name_pred], errors='coerce')
            temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            temp_df.dropna(inplace=True)

            if len(temp_df) > 0:
                margin_measures_auroc_optimal_temperature = sklearn.metrics.roc_auc_score(
                    1 - temp_df['correct'],
                    temp_df[col_name_pred]
                )
                result_dict[f'margin_measures_T{best_temp}_auroc'] = margin_measures_auroc_optimal_temperature 
                print(f"Success: {col_name_pred} AUROC = {margin_measures_auroc_optimal_temperature:.4f}")
            else:
                print(f"Warning: No valid samples left for {col_name_pred} after cleaning.")
        else:
            print(f"Warning: Column {col_name_pred} not found in DataFrame.")

        col_name_pred = f'unnormalised_margin_measures_T{best_temp}'

        if col_name_pred in result_df_credal.columns:
            temp_df = result_df_credal[['correct', col_name_pred]].copy()

            # Same 3-step cleaning process
            temp_df[col_name_pred] = temp_df[col_name_pred].apply(safe_to_float)
            temp_df[col_name_pred] = pd.to_numeric(temp_df[col_name_pred], errors='coerce')
            temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            temp_df.dropna(inplace=True)

            if len(temp_df) > 0:
                unnormalised_margin_measures_auroc_optimal_temperature = sklearn.metrics.roc_auc_score(
                    1 - temp_df['correct'],
                    temp_df[col_name_pred]
                )
                result_dict[f'unnormalised_margin_measures_T{best_temp}_auroc'] = unnormalised_margin_measures_auroc_optimal_temperature 
                print(f"Success: {col_name_pred} AUROC = {unnormalised_margin_measures_auroc_optimal_temperature:.4f}")
            else:
                print(f"Warning: No valid samples left for {col_name_pred} after cleaning.")
        else:
            print(f"Warning: Column {col_name_pred} not found in DataFrame.")

        col_name_pred = f'mutual_information_T{best_temp}'

        if col_name_pred in result_df_credal.columns:
            temp_df = result_df_credal[['correct', col_name_pred]].copy()

            # Same 3-step cleaning process
            temp_df[col_name_pred] = temp_df[col_name_pred].apply(safe_to_float)
            temp_df[col_name_pred] = pd.to_numeric(temp_df[col_name_pred], errors='coerce')
            temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            temp_df.dropna(inplace=True)

            if len(temp_df) > 0:
                mutual_information_auroc_optimal_temperature = sklearn.metrics.roc_auc_score(
                    1 - temp_df['correct'],
                    temp_df[col_name_pred]
                )
                result_dict[f'mutual_information_T{best_temp}_auroc'] = mutual_information_auroc_optimal_temperature 
                print(f"Success: {col_name_pred} AUROC = {mutual_information_auroc_optimal_temperature:.4f}")
            else:
                print(f"Warning: No valid samples left for {col_name_pred} after cleaning.")
        else:
            print(f"Warning: Column {col_name_pred} not found in DataFrame.")

        col_name_pred = f'ln_predictive_entropy_T{best_temp}'

        if col_name_pred in result_df_credal.columns:
            temp_df = result_df_credal[['correct', col_name_pred]].copy()

            # Same 3-step cleaning process
            temp_df[col_name_pred] = temp_df[col_name_pred].apply(safe_to_float)
            temp_df[col_name_pred] = pd.to_numeric(temp_df[col_name_pred], errors='coerce')
            temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            temp_df.dropna(inplace=True)

            if len(temp_df) > 0:
                ln_predictive_entropy_auroc_optimal_temperature = sklearn.metrics.roc_auc_score(
                    1 - temp_df['correct'],
                    temp_df[col_name_pred]
                )
                result_dict[f'mutual_information_T{best_temp}_auroc'] = ln_predictive_entropy_auroc_optimal_temperature 
                print(f"Success: {col_name_pred} AUROC = {ln_predictive_entropy_auroc_optimal_temperature:.4f}")
            else:
                print(f"Warning: No valid samples left for {col_name_pred} after cleaning.")
        else:
            print(f"Warning: Column {col_name_pred} not found in DataFrame.")

    else:
        print(f"FAILED TO LOAD {opt_temp_path}")

    result_dict[f'average_rougeL_auroc_T{best_temp}'] = sklearn.metrics.roc_auc_score(result_df_credal['correct'],
                                                                        result_df_credal['rougeL_among_generations_credal'])

    p_true_path = f'{path_prefix}{model_name}_p_true_aurocs.pkl'
    if os.path.exists(p_true_path):
        try:
            with open(p_true_path, 'rb') as f:
                p_true_auroc = pickle.load(f)
            
            # If it's a tensor, convert to float
            if hasattr(p_true_auroc, 'item'):
                p_true_auroc = p_true_auroc.item()
                
            result_dict['p_true_auroc'] = float(p_true_auroc)
            print(f"Loaded p_true AUROC: {p_true_auroc:.4f}")
        except Exception as e:
            print(f"Warning: Failed to load or parse p_true file: {e}")
    else:
        print(f"p_true file not found: {p_true_path}")

    if args.verbose:
        print('Number of samples:', len(result_df))
        print(result_df['predictive_entropy'].mean())
        print(result_df['average_predictive_entropy'].mean())
        print(result_df['predictive_entropy_over_concepts'].mean())
        print('ln_predictive_entropy_auroc', ln_predictive_entropy_auroc)
        print('semantci entropy auroc', entropy_over_concepts_auroc)
        print(
            'Semantic entropy +',
            sklearn.metrics.roc_auc_score(
                1 - result_df['correct'],
                result_df['predictive_entropy_over_concepts'] - 3 * result_df['rougeL_among_generations']))
        print('RougeL among generations auroc',
              sklearn.metrics.roc_auc_score(result_df['correct'], result_df['rougeL_among_generations']))
        print('margin measure auroc:', result_dict['margin_measure_auroc'])

    # Measure the AURROCs when using different numbers of generations to compute our uncertainty measures.
    ln_aurocs = []
    aurocs = []
    semantic_aurocs = []
    average_number_of_semantic_sets = []
    average_number_of_semantic_sets_correct = []
    average_number_of_semantic_sets_incorrect = []
    for i in range(1, num_generations + 1):
        ln_predictive_entropy_auroc = sklearn.metrics.roc_auc_score(
            1 - result_df['correct'], result_df['average_predictive_entropy_on_subset_{}'.format(i)])
        aurocs.append(
            sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                          result_df['predictive_entropy_on_subset_{}'.format(i)]))
        ln_aurocs.append(ln_predictive_entropy_auroc)
        semantic_aurocs.append(
            sklearn.metrics.roc_auc_score(1 - result_df['correct'],
                                          result_df['semantic_predictive_entropy_on_subset_{}'.format(i)]))
        average_number_of_semantic_sets.append(result_df['number_of_semantic_sets_on_subset_{}'.format(i)].mean())
        average_number_of_semantic_sets_correct.append(
            result_df[result_df['correct'] == 1]['number_of_semantic_sets_on_subset_{}'.format(i)].mean())
        average_number_of_semantic_sets_incorrect.append(
            result_df[result_df['correct'] == 0]['number_of_semantic_sets_on_subset_{}'.format(i)].mean())

    result_dict['ln_predictive_entropy_auroc_on_subsets'] = ln_aurocs
    result_dict['predictive_entropy_auroc_on_subsets'] = aurocs
    result_dict['semantic_predictive_entropy_auroc_on_subsets'] = semantic_aurocs
    result_dict['average_number_of_semantic_sets_on_subsets'] = average_number_of_semantic_sets
    result_dict['average_number_of_semantic_sets_on_subsets_correct'] = average_number_of_semantic_sets_correct
    result_dict['average_number_of_semantic_sets_on_subsets_incorrect'] = average_number_of_semantic_sets_incorrect
    result_dict['model_name'] = model_name
    result_dict['run_name'] = run_name

    wandb.log(result_dict)

    overall_result_dict[run_id] = result_dict
    sequence_embeddings_dict[run_id] = sequence_embeddings

    wandb.finish()
    torch.cuda.empty_cache()

with open('overall_results.json', 'w') as f:
    json.dump(overall_result_dict, f)

with open('sequence_embeddings.pkl', 'wb') as f:
    pickle.dump(sequence_embeddings_dict, f)

# Store data frame as csv
accuracy_verification_df = result_df[['most_likely_generation', 'answer', 'correct']]
accuracy_verification_df.to_csv('accuracy_verification.csv')

