from transformers import AutoModelForCausalLM, AutoTokenizer
import config 
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", cache_dir=config.hf_cache_dir)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", cache_dir=config.hf_cache_dir)

