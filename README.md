# Credal Prediction for Large Language Models (LLMs)

 A framework for robust uncertainty quantification in Large Language Models using Credal Sets.

## Overview

Standard LLMs output a single probability distribution over the vocabulary for the next token. This approach conflates 
**aleatoric uncertainty** (inherent ambiguity in language) with **epistemic uncertainty** 
(the model's actual lack of knowledge). 

This repository implements **Credal Prediction for LLMs**, a framework that outputs a set of probability distributions 
(Credal Set) rather than a single distribution over multiple different temperature values.

##  Installation

Clone the repository and install the dependencies (thesis-env.yml). We recommend using a virtual environment.

```bash
git clone [https://github.com/liamhpr/Credal-Prediction-for-LLMs.git](https://github.com/liamhpr/Credal-Prediction-for-LLMs.git)
cd Credal-Prediction-for-LLMs
