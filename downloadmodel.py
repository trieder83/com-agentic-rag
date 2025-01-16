from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model name and local directory
model_name = "qualcomm/Mistral-3B"
local_dir = "model/mistral-3b"

# Download the model and tokenizer to the specified directory
print("Downloading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)

print(f"Model and tokenizer downloaded to: {local_dir}")

