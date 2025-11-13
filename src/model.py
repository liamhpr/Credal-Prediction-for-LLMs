import os

import config
import argparse
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='coqa')
args = parser.parse_args()

seed_value = 10


os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

model = AutoModelForCausalLM.from_pretrained(f"facebook/{args.model}",
                                             torch_dtype=torch.float16,
                                             cache_dir=config.hf_cache_dir).cuda()

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)

if args.dataset == 'coqa':
    dataset = datasets.load_from_disk('data/sets/coqa_dev_v1.json')
    id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
elif args.dataset == 'trivia_qa':
    # TODO
    pass
