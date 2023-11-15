#syntax=docker/dockerfile:experimental

# stage 1
# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as base
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 as base

ENV COMMAND=""

# Model Related Information
ENV SAKURA_MODEL_NAME="Sakura-13B-LNovel-v0_8-4bit"
ENV SAKURA_MODEL_VERSION="0.8"

# Server Related Information
ENV USERNAME="sakura"
ENV PASSWORD="sakura"
ENV ADDRESS="127.0.0.1"
ENV PORT="5000"
ENV LOGLEVEL="info"

ENV MODEL_ROOT=/models
ENV SAKURA_MODEL_PATH=${MODEL_ROOT}/${SAKURA_MODEL_NAME}

LABEL sakura_model.name=${SAKURA_MODEL_NAME}
LABEL sakura_model.version=${SAKURA_MODEL_VERSION}

LABEL moe.kuriko.release-date="2023-11-09"
LABEL moe.kuriko.version=${SAKURA_MODEL_VERSION}
LABEL mode.kuriko.commit=${GIT_COMMIT}

EXPOSE ${PORT}

WORKDIR /work

# install python and essential packages
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y \
        python3=3.10.* python3-pip \
        build-essential curl wget gcc 

# stage 2
FROM base as pip

COPY ./requirements.txt /work/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt


# stage 3
FROM pip as working

COPY . .
RUN mkdir -p ${SAKURA_MODEL_PATH}

ENTRYPOINT ["/bin/bash", "./docker/scripts/run.sh"]