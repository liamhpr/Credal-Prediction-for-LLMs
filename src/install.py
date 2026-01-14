from transformers import AutoTokenizer, OPTForCausalLM
import os

# CHANGED: Point to the local directory where you cloned the repo
# Ensure the path is correct relative to where you run the script
model_path = "/dss/dsshome1/03/ra54sov2/Credal-Prediction-for-LLMs/src/hf_dir/opt-6.7b"

if os.path.exists(model_path):
    print(f"Loading from local path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = OPTForCausalLM.from_pretrained(model_path)
else:
    print("Error: Local model path not found. Please clone the repo first.")
    exit()

print("Model loaded successfully!")
