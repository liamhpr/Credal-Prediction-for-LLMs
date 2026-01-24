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

model='opt-6.7b'

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

echo 'Starting get_prompting_based_uncertainty.py';
python get_prompting_based_uncertainty.py \
		   --run_id_for_few_shot_prompt=$run_id \
		   --run_id_for_evaluation=$run_id;

echo 'Starting compute_confidence_measure.py';
python compute_confidence_measure.py \
		   --generation_model=$model \
		   --evaluation_model=$model \
		   --run_id=$run_id;

echo 'Pipeline finished successfully';
"
