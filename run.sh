#!/bin/bash

#docker run -it --rm -p --name embedding embedding
volume=$PWD/model
#faissvol=$PWD/faiss_indexes

if [ -x "$(command -v docker)" ]; then
EXEC=docker
IMGAGE_PREFIX=
else
EXEC=podman
IMAGE_PREFIX=localhost/
fi
DATA=$PWD/data
TICKTOKEN=$PWD/tiktoken
APP=$PWD/src

if [ ! -z $WINDIR ]; then
  volume=c:\\git\\a\\com-agentic-rag\\model
  #faissvol=c:\\git\\a\\com-agentic-rag\\faiss_indexes
  CACHE=c:\\Users\\${USERNAME}\\.cache\\huggingface\\hub
  DOCKER="winpty podman"
  DATA=c:\\git\\a\\com-agentic-rag\\data
  TICKTOKEN=c:\\git\\a\\com-agentic-rag\\tiktoken
  APP=c:\\git\\a\\com-agentic-rag\\src
fi
#$EXEC run -p 8888:8888 -it embeddings:latest flask --app createEmbeddings2 run --host 0.0.0.0 --port 8888 -v $volume:/data
#$EXEC run -p 8888:8888  --rm -v $volume:/model -v $faissvol:/faiss_indexes localhost/trieder83/com-agentic-rag:0.1  flask --app agenticrag.py  run --host 0.0.0.0 --port 8888 
#$EXEC run -p 8888:8888  --rm -v $CACHE:/cache localhost/trieder83/com-agentic-rag:0.1  #python test.py #flask --app agenticrag.py  run --host 0.0.0.0 --port 8888 
#$EXEC run --rm  -v $PWD/data:/data -v $PWD/src:/app -v $PWD/tiktoken:/tiktoken -env TIKTOKEN_CACHE_DIR=/tiktoken --network="host" ${IMGAGE_PREFIX}trieder83/com-agentic-rag:0.1
echo $IMAGE_PREFIX
#echo $EXEC run --rm  -v $PWD/data:/data -v $PWD/src:/app -v $PWD/tiktoken:/tiktoken --env "TIKTOKEN_CACHE_DIR=/tiktoken" --network="host" localhost/trieder83/com-agentic-rag:0.1
#$EXEC run --rm -v $PWD/data:/data -v $PWD/src:/app -v $PWD/tiktoken:/tiktoken --network="host" localhost/trieder83/com-agentic-rag:0.1

#$EXEC run --rm --name notebook -v $DATA:/data -v $APP:/app -v $TICKTOKEN:/tiktoken --env-file=.env --env "TIKTOKEN_CACHE_DIR=/tiktoken" --network="host" ${IMAGE_PREFIX}trieder83/com-agentic-rag:0.2
$EXEC run --rm --name notebook -v $DATA:/data -v $APP:/app -v $TICKTOKEN:/tiktoken --env-file=.env --env "TIKTOKEN_CACHE_DIR=/tiktoken" --net ainet -p 8888:8888 ${IMAGE_PREFIX}trieder83/com-agentic-rag:0.2

# podman network create ainet
# podman run -d --gpus=all -v ollama:/c/temp/ollama -p 11434:11434 --name ollama ollama/ollama
