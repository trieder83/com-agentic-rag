#!/bin/bash

#docker run -it --rm -p --name embedding embedding
volume=$PWD/model
#faissvol=$PWD/faiss_indexes

if [ -x "$(command -v docker)" ]; then
EXEC=docker
else
EXEC=podman
fi

if [ ! -z $WINDIR ]; then
  volume=c:\\git\\a\\com-agentic-rag\\model
  #faissvol=c:\\git\\a\\com-agentic-rag\\faiss_indexes
  CACHE=c:\\Users\\${USERNAME}\\.cache\\huggingface\\hub
  DOCKER="winpty podman"
fi
#$EXEC run -p 8888:8888 -it embeddings:latest flask --app createEmbeddings2 run --host 0.0.0.0 --port 8888 -v $volume:/data
#$EXEC run -p 8888:8888  --rm -v $volume:/model -v $faissvol:/faiss_indexes localhost/trieder83/com-agentic-rag:0.1  flask --app agenticrag.py  run --host 0.0.0.0 --port 8888 
#$EXEC run -p 8888:8888  --rm -v $CACHE:/cache localhost/trieder83/com-agentic-rag:0.1  #python test.py #flask --app agenticrag.py  run --host 0.0.0.0 --port 8888 
$EXEC run --rm  -v $PWD/data:/data -v $PWD/src:/app --network="host" trieder83/com-agentic-rag:0.1

