#!/bin/bash


if [ -x "$(command -v docker)" ]; then
EXEC=docker
else
EXEC=podman
fi
$EXEC  build --tag 'trieder83/com-agentic-rag:0.1' .
