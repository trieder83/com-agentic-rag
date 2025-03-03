# FROM python:3
FROM python:3.13-slim as build
#FROM python:3.11
#FROM pytorch/pytorch
#FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04
#FROM nvidia/cuda:12.6.2-base-ubuntu22.04
#FROM jupyter/base-notebook:x86_64-python-3.11.6


# Set environment variables.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt update && \
    apt install --no-install-recommends -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# Set environment variables.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONIOENCODING=utf-8

#    apt install -y python3-numpy
#  apt install -y g++
#    apt install -y build-essentials
#  apt install -y pip python3-setuptools python3-distutils-extra

FROM python:3.13-slim as runtime
#RUN apt-get update && \
#    apt-get install -y python3-pip python3-dev && \
#    rm -rf /var/lib/apt/lists/* && \
RUN mkdir /app && \
    mkdir /app/.local && \
    chown 1001:1001 /app /app/.local

# copy from build image
COPY --chown=1001:1002 --from=build /opt/venv /opt/venv


#RUN apt-get update && apt-get install --no-install-recommends -y tk \
#    && rm -rf /var/lib/apt/lists/* 

WORKDIR /app
USER 1001:1001

# Path - this activates the venv
ENV PATH="/opt/venv/bin:$PATH"

ENV PORT 8888
ENV HOST 0.0.0.0
ENV JUPYTER_ALLOW_INSECURE_WRITES=1
ENV HF_HOME="/cache"

#RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch && \
#RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \

#&& \
#  python3 -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./app/artefacts")'
#RUN python3 ./createEmbeddings2.py
#RUN python3 ./embeddingsolr.py
#    huggingface-cli login --token xxx && \
#COPY --exclude=data* --exclude=model agenticrag.py .
#COPY --exclude=data* --exclude=model . .

#RUN [ "python3", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')" ]

COPY src/ /app/
COPY notebooks/ /app/notebooks/

#CMD [ "python", "./test3.py" ]
#CMD [ flask, --app, createEmbeddings2, "run"]
#CMD [ flask, --app, agenticrag.py, "run"]
#CMD [ ./"bootstrap.sh"]
EXPOSE 8888
#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD [ "jupyter","notebook","--ip=0.0.0.0", "--no-browser", "--allow-root", --notebook-dir=/app,"--NotebookApp.default_url=/app/notebooks/default.ipynb"]
