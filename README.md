Init


# test / dev
 python agenticrag.py  --rmindex Y --query "america"


# ollama (win)
podman run -d --gpus=all -v ollama:/c/temp/ollama -p 11434:11434 --name ollama ollama/ollama
podman exec 3a5a5bf96c3f ollama pull odellama:7b-code
podman cp 3a5a5bf96c3f:/root/.ollama dot_ollama
