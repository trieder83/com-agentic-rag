from huggingface_hub import snapshot_download

# Download the entire repository
repo_id = "meta-llama/Llama-3.2-1B"
local_dir = "model/llama-3.2-1B"

snapshot_download(repo_id, local_dir=local_dir)
print(f"Model downloaded to: {local_dir}")

