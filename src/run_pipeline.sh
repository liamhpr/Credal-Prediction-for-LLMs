#!/bin/bash
#SBATCH --clusters=hlai
#SBATCH --gpus=1
#SBATCH --job-name="credal_prediction_for_llms"

source ~/.bashrc
conda activate thesis-env

# Generate a run ID
run_id=$(python - << 'EOF'
import wandb
rid = wandb.util.generate_id()
print(rid)
EOF
)

model='opt-350m'

echo "Using run_id: $run_id"
echo "Using model: $model"

srun bash -c "
set -e;

echo 'Starting generate.py';
python generate.py --num_generations_per_prompt 5 \
                   --model $model \
                   --fraction_of_data_to_use 0.4 \
                   --run_id $run_id \
                   --temperature 0.5 \
                   --num_beams 1 \
                   --top_p 1.0;

echo 'Starting clean_generated_string.py';
python clean_generated_string.py \
                   --generation_model $model \
                   --run_id $run_id;

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


echo 'Starting generate.py for test split';
python generate.py --num_generations_per_prompt 5 \
                   --model $model \
                   --fraction_of_data_to_use 0.02 \
                   --run_id $run_id \
                   --temperature 0.5 \
                   --num_beams 1 \
                   --top_p 1.0 \
		   --use_test_split;

echo 'Starting clean_generated_string.py for test split';
python clean_generated_string.py \
                   --generation_model $model \
                   --run_id $run_id \
		   --use_test_split;

echo 'Starting get_semantic_similarities.py for test split';
python get_semantic_similarities.py \
                   --generation_model $model \
                   --run_id $run_id \
		   --use_test_split;

echo 'Starting get_likelihoods.py for test split';
python get_likelihoods.py \
                   --evaluation_model $model \
                   --generation_model $model \
                   --run_id $run_id \
		   --use_test_split;

echo 'Pipeline finished successfully';
"
