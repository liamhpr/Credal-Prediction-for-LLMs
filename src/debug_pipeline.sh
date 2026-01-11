#!/bin/bash
#SBATCH --clusters=hlai
#SBATCH --gpus=1
#SBATCH --job-name="credal_prediction_for_llms"

source ~/.bashrc
conda activate thesis-env

# Generate a run ID
run_id='ji6z5yys'

model='opt-350m'

echo "Using run_id: $run_id"
echo "Using model: $model"

srun bash -c "
set -e;

echo 'Starting get_semantic_similarities.py';
python get_semantic_similarities.py \
                   --generation_model $model \
                   --run_id $run_id;

echo 'Starting get_likelihoods.py';
python get_likelihoods.py \
                   --evaluation_model $model \
                   --generation_model $model \
                   --run_id $run_id;

echo 'Pipeline finished successfully';
"
