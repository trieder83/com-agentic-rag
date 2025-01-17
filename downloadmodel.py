from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
login()

# Define the model name and local directory
#model_name = "qualcomm/Mistral-3B"
#local_dir = "model/mistral-3b"
#model_name="meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8"
model_name="meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8"
#model_name = "meta-llama/Llama-3.2-1B"
local_dir = "model/llama-3.2-1B_instruct_qint4"
save_dir= "model/llama-3.2-1B_instruct_qint4_local"

# Download the model and tokenizer to the specified directory
print("Downloading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)

print(f"Model and tokenizer downloaded to: {local_dir}")

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

