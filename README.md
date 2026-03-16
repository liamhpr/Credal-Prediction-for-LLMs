# Credal Prediction for Large Language Models (LLMs)

 A framework for robust uncertainty quantification in Large Language Models using Credal Sets.

## Overview

Standard LLMs output a single probability distribution over the vocabulary for the next token. This approach conflates 
**aleatoric uncertainty** (inherent ambiguity in language) with **epistemic uncertainty** 
(the model's actual lack of knowledge). 

This repository implements **Credal Prediction for LLMs**, a framework that outputs a set of probability distributions 
(Credal Set) rather than a single distribution over multiple different temperature values.

# Running the Full Pipeline

To execute the entire workflow, we rely on a SLURM batch script. You can trigger the full pipeline by 
submitting the script to your cluster:
```bash
sbatch run_pipeline.sh
```

# Setup and Configuration
Before running the pipeline, you will need to set up your data and environment:

Data Preprocessing: The parse_coqa.py script handles downloading the dataset from Hugging Face, 
tokenizing the text, and saving the processed files locally. You only need to run these scripts once.

Directory Configuration: Open config.py to map out your preferred file paths for saving both intermediate checkpoints 
and final results.

Environment Setup: All necessary dependencies are tracked in the provided thesis-env.yml file, which you can use to 
replicate our conda environment.

# Generating Answers and Computing Uncertainty Measures
The pipeline breaks down the generation and uncertainty quantification into the following steps:

```generate.py```: Produces and stores for every temperature in the temperatuer-interval [0.4, 2.2]
a batch of answers for a targeted subset of questions and evaluates the base question-answering accuracy of those 
generations.

```clean_generated_string.py```: Cleans the raw outputs to remove irrelevant trailing text—for instance, pruning 
instances where the model answers the prompt and then hallucinates a follow-up question.

```get_semantic_similarities.py```: Analyzes the cleaned answers across all valid temperatures and groups them into 
semantic clusters based on their meaning.

```get_prompting_based_uncertainty.py```: Evaluates and establishes the p(True) baseline metric.

```compute_likelihoods.py```: Determines the precise likelihood of each generated response as 
scored by the generating model.

```compute_confidence_measure.py```: Calculates a comprehensive suite of uncertainty and confidence metrics, 
including our credal semantic entropy, semantic entropy, predictive entropy, lexical similarity, and p(True).

Once the full pipeline has successfully executed, you can evaluate the outcomes using ```analyze_result.py```. 
This script processes the final data to generate performance metrics, including the AUROC score.

##  Installation

Clone the repository and install the dependencies (thesis-env.yml). We recommend using a virtual environment.

```bash
git clone [https://github.com/liamhpr/Credal-Prediction-for-LLMs.git](https://github.com/liamhpr/Credal-Prediction-for-LLMs.git)
cd Credal-Prediction-for-LLMs
```
