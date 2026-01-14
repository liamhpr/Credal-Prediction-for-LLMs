import argparse
import os
import pickle
import random

import numpy as np
import torch
import utils.utils as utils
import logging

import wandb
import config

utils.setup_logger()
logging.info('Starting get_likelihoods.py...')

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--evaluation_model', type=str, default='opt-350m')
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


#wandb.init(project='credal-prediction-for-large-language-models', id=args.run_id, config=args, resume='allow')

run_name = wandb.run.name


def classwise_adding_optim_logit(logits_train, targets_train, logits_test, n_classes):
    csets = []
    rls = []
    mll = log_likelihood(logits_train, targets_train).cpu().detach().item()
    for alpha in tqdm(alphas, desc='Alphas'):
        bounds = []
        for k in range(n_classes):
            # 1 is finding minimum, -1 is finding maximum
            bound = []
            for direction in [1, -1]:
                def fun(x):
                    return direction * x[k]
                    # c = torch.tensor(x, device=logits_train.device)
                    # logits_train_T = logits_train + c
                    # probs = F.softmax(logits_train_T, dim=1).cpu().detach().numpy()
                    # return direction * np.mean(probs[:, k], axis=0)

                def const(x) -> float:
                    c = torch.tensor(x, device=logits_train.device)
                    logits_train_T = logits_train + c
                    lik = log_likelihood(logits_train_T, targets_train).cpu().detach().item()
                    rel_lik = np.exp(lik - mll)
                    return rel_lik

                x0 = np.zeros(n_classes)
                optim_bounds = [(0.0, 0.0)] * n_classes
                # T_abs = int(np.max((abs(torch.max(logits_train).cpu().numpy()), abs(torch.min(logits_train).cpu().numpy()))))
                # optim_bounds[k] = (-2 * T_abs, 2 * T_abs)
                optim_bounds[k] = (None, None)
                constraints = {'type': 'ineq', 'fun': lambda x: const(x) - alpha}
                res = minimize(fun, x0, constraints=constraints, bounds=optim_bounds)
                bound.append(res.x)
            bounds.append(bound)

        # add the bounds to the logits_test to make predictions
        for k in range(n_classes):
            # both ``directions''
            for d in range(2):
                logits_test_T = logits_test + torch.tensor(bounds[k][d], device=logits_test.device)
                csets.append(F.softmax(logits_test_T, dim=1).cpu().detach().numpy())
        rls.append([alpha] * (2 * n_classes))
    csets = np.array(csets)
    rls = np.array(rls).flatten()
    return csets, rls


output_dir = config.output_dir
with open(f'{output_dir}sequences/{run_name}/train_split/{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl', 'rb') as f:
    train_results = pickle.load(f)

# WARNING: Padding is missing
logits_train = torch.stack([
    r['cluster_log_likelihoods'] for r in train_results
])

# WARNING: How should I compute targets_train??? https://www.notion.so/How-to-define-the-targets_train-2e7dea560e8f8018841ce40681f4a147
targets_train = torch.argmax(logits_train, dim=1)


with open(f'{output_dir}sequences/{run_name}/test_split/{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl', 'rb') as f:
    test_results = pickle.load(f)

logits_test = torch.stack([
    r['cluster_log_likelihoods'] for r in test_results
])


csets, rls = classwise_adding_optim_logit(logits_train, targets_train, logits_test, args.classes)
with open(f'{output_dir}sequences/{run_name}/{args.generation_model}_credalsets.pkl', 'wb') as outfile: 
    pickle.dump(csets, outfile)
