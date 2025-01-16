from huggingface_hub import snapshot_download

# Download the entire repository
repo_id = "qualcomm/Mistral-3B"
local_dir = "model/mistral-3b"

snapshot_download(repo_id, local_dir=local_dir)
print(f"Model downloaded to: {local_dir}")

