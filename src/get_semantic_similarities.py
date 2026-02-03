import argparse
import csv
import os
import pickle
import random
import logging

import utils.utils as utils
import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
import wandb

utils.setup_logger()
logging.info('Starting get_semantic_similarities.py...')

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

model_path = config.get_model_path(args.generation_model)
generation_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, cache_dir=config.data_dir)

tokenizer = AutoTokenizer.from_pretrained("/dss/dsshome1/03/ra54sov2/Credal-Prediction-for-LLMs/src/hf_dir/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("/dss/dsshome1/03/ra54sov2/Credal-Prediction-for-LLMs/src/hf_dir/deberta-large-mnli").cuda()

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

input_file = f'{path_prefix}{args.generation_model}_all_generations.pkl'
with open(input_file, 'rb') as infile:
    all_temperature_sequences = pickle.load(infile)


# ============================= STEP 1 AGGREGATIOIN ===============================
# Collect all unique answers for each question ID across ALL temperatures
logging.info("Aggregating unique answers across all temperatures...")

global_question_map = {}
# Structre: {
#    question_id: {
#        'question': str,
#        'unique_texts': set(str),
#        'cluster_map': dict() <- to be filled later
#    }
#}

for temp, samples in all_temperature_sequences.items():
    for sample in samples:
        q_id = sample['id'][0] if isinstance(sample['id'], list) else sample['id']

        if q_id not in global_question_map:
            global_question_map[q_id] = {
                'question': sample['question'],
                'unique_texts': set()
            }

        # Get texts (prefer cleaned if available)
        if 'cleaned_generated_texts' in sample:
            texts = sample['cleaned_generated_texts']
        else:
            texts = sample['generated_texts']

        global_question_map[q_id]['unique_texts'].update(texts)


# ======================= STEP 2: GLOBAL CLUSTERING ============================ 
# Perform NLI comparison on the unique set for each question 
logging.info(f"Clustering answers for {len(global_question_map)} unique questions...")

deberta_predictions = []

#meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')
rouge_types = ['rouge1', 'rouge2', 'rougeL']

for q_id, data in tqdm(global_question_map.items()):
    question = data['question']
    unique_generated_texts = list(data['unique_texts'])

    # Initialize: Every answer is its owon cluster initially 
    # Map: text -> cluster_id
    semantic_set_ids = {text: i for i, text in enumerate(unique_generated_texts)}

    # Only run NLI if we have more than 1 unique answer
    if len(unique_generated_texts) > 1: 
        # Compare every pair (O(N^2))
        for i, text_i in enumerate(unique_generated_texts):
            for j in range(i + 1, len(unique_generated_texts)):
                text_j = unique_generated_texts[j]

                qa_1 = f"{question} {text_i}"
                qa_2 = f"{question} {text_j}"

                # Forward comparison
                input_seq = f"{qa_1} [SEP] {qa_2}"
                encoded_input = tokenizer.encode(input_seq, padding=True, return_tensors='pt').to(device)
                logits = model(encoded_input)['logits']
                predicted_label = torch.argmax(logits, dim=1)

                # Reverse comparison
                reverse_input_seq =f"{qa_2} [SEP] {qa_1}"
                encoded_reverse = tokenizer.encode(reverse_input_seq, padding=True, return_tensors='pt').to(device)
                reverse_logits = model(encoded_reverse)['logits']
                reverse_predicted_label = torch.argmax(reverse_logits, dim=1)

                deberta_prediction = 1
                print(qa_1, qa_2, predicted_label, reverse_predicted_label)
                if 0 in predicted_label or 0 in reverse_predicted_label:
                    has_semantically_different_answers = True
                    deberta_prediction = 0

                else:
                    semantic_set_ids[text_j] = semantic_set_ids[text_i]

                deberta_predictions.append([text_i, text_j, deberta_prediction])
    

    final_ids = sorted(list(set(semantic_set_ids.values())))
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(final_ids)}

    final_cluster_map = {text: id_mapping[sid] for text, sid in semantic_set_ids.items()}
    global_question_map[q_id]['cluster_map'] = final_cluster_map

# ========================= STEP 3: ASSIGNMENT & SYNTACTIC CALC (Per Sample) =====================================
logging.info("Assigning Cluster IDs back to temperature samples...")

# We will safe the FULL dictionary with the new fields added
# This preserves the {temp: [samples]} structure for the next script.

for temp, samples in all_temperature_sequences.items():
    for sample in tqdm(samples, desc=f"Temp {temp}"):
        q_id = sample['id'][0] if isinstance(sample['id'], list) else sample['id']

        if 'cleaned_generated_texts' in sample:
            texts = sample['cleaned_generated_texts']
        else:
            texts = sample['generated_texts']

        # 1. Assign Semantic IDs using the GLOBAL map
        # This guarantees consistency across temperatures
        cluster_map = global_question_map[q_id]['cluster_map']
        sample['semantic_set_ids'] = [cluster_map[t] for t in texts]

        # 2. Compute Syntactic Similarity (ROUGE)
        # This is specific to the batch generated at this temperature
        # (It measures diversity within this specific generation set)
        syntactic_similarities = {rtype: 0.0 for rtype in rouge_types}
        
        if len(texts) > 1: 
            # Creeate pairs for ROUGE (all-vs-all within this sample)
            preds = []
            refs = []
            for i in texts: 
                for j in texts:
                    if i != j:
                        preds.append(i)
                        refs.append(j)

            if preds:
                results = rouge.compute(predictions=preds, references=refs)
                for rtype in rouge_types:
                    syntactic_similarities[rtype] = results[rtype]


        sample['syntactic_similarities'] = syntactic_similarities

        # Flag if this specific sample produced semantically different answers
        # (i.e., did this temperature produce > 1 cluster?)
        unique_clusters_in_sample = set(sample['semantic_set_ids'])
        sample['has_semantically_different_answers'] = len(unique_clusters_in_sample) > 1

# ================================== STEP 4: SAVE ===========================================
with open(f'{path_prefix}deberta_predictions_{args.run_id}.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['qa_1', 'qa_2', 'predictions'])
    writer.writerows(deberta_predictions)

output_file = f'{path_prefix}{args.generation_model}_generations_similarities.pkl'
with open(output_file, 'wb') as outfile:
    pickle.dump(all_temperature_sequences, outfile)

logging.info(f"Finished. Saved processed data to {output_file}")







"""
for sample in tqdm(sequences):
    question = sample['question']
    if 'cleaned_generated_texts' in sample:
        generated_texts = sample['cleaned_generated_texts']
    else:
        generated_texts = sample['generated_texts']

    id_ = sample['id'][0]

    unique_generated_texts = list(set(generated_texts))

    answer_list_1 = []
    answer_list_2 = []
    has_semantically_different_answers = False
    inputs = []
    syntactic_similarities = {}
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    for rouge_type in rouge_types:
        syntactic_similarities[rouge_type] = 0.0

    semantic_set_ids = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index

    print('Number of unique answers:', len(unique_generated_texts))

    if len(unique_generated_texts) > 1:

        # Evalauate semantic similarity
        for i, reference_answer in enumerate(unique_generated_texts):
            for j in range(i + 1, len(unique_generated_texts)):

                answer_list_1.append(unique_generated_texts[i])
                answer_list_2.append(unique_generated_texts[j])

                qa_1 = question + ' ' + unique_generated_texts[i]
                qa_2 = question + ' ' + unique_generated_texts[j]

                input = qa_1 + ' [SEP] ' + qa_2
                inputs.append(input)
                encoded_input = tokenizer.encode(input, padding=True)
                prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                predicted_label = torch.argmax(prediction, dim=1)

                reverse_input = qa_2 + ' [SEP] ' + qa_1
                encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                deberta_prediction = 1
                print(qa_1, qa_2, predicted_label, reverse_predicted_label)
                if 0 in predicted_label or 0 in reverse_predicted_label:
                    has_semantically_different_answers = True
                    deberta_prediction = 0

                else:
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]

                deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])

        rouge = evaluate.load('rouge')

        # Evalauate syntactic similarity
        answer_list_1 = []
        answer_list_2 = []
        for i in generated_texts:
            for j in generated_texts:
                if i != j:
                    answer_list_1.append(i)
                    answer_list_2.append(j)

        results = rouge.compute(predictions=answer_list_1, references=answer_list_2)

        for rouge_type in rouge_types:
            syntactic_similarities[rouge_type] = results[rouge_type]

    result_dict[id_] = {
        'syntactic_similarities': syntactic_similarities,
        'has_semantically_different_answers': has_semantically_different_answers
    }
    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
    result_dict[id_]['semantic_set_ids'] = list_of_semantic_set_ids

with open('{}deberta_predictions_{}.csv'.format(path_prefix, args.run_id), 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(['qa_1', 'qa_2', 'prediction'])
    writer.writerows(deberta_predictions)

print(result_dict)

    
with open(f'{path_prefix}{args.generation_model}_generations_similarities.pkl', 'wb') as outfile:
    pickle.dump(result_dict, outfile)
"""
