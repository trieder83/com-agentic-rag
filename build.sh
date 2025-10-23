#!/bin/bash


export TIKTOKEN_CACHE_DIR=./tiktoken
if [ -x "$(command -v docker)" ]; then
EXEC=docker
else
EXEC=podman
fi
#$EXEC  build --build-arg http_proxy=$HTTP_PROXY --build-arg https_proxy=$HTTPS_PROXY --tag 'trieder83/com-agentic-rag:0.2' . 
$EXEC  build --no-cache --network=host --tag 'trieder83/com-agentic-rag:0.7' .
