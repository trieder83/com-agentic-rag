#!/bin/bash

#docker run -it --rm -p --name embedding embedding
volume=$PWD/model
faissvol=$PWD/faiss_indexes
DOCKER=docker

if [ ! -z $WINDIR ]; then
  volume=c:\\git\\a\\com-agentic-rag\\model
  faissvol=c:\\git\\a\\com-agentic-rag\\faiss_indexes
  DOCKER="winpty podman"
fi
#docker run -p 8888:8888 -it embeddings:latest flask --app createEmbeddings2 run --host 0.0.0.0 --port 8888 -v $volume:/data
$DOCKER run -p 8888:8888  --rm -v $volume:/model -v $faissvol:/faiss_indexes localhost/trieder83/com-agentic-rag:0.1  flask --app agenticrag.py  run --host 0.0.0.0 --port 8888 

