#!/bin/bash

ORGNIZATION="kurikomoe"
NAME="sakura-13b-galgame"
TAG="0.8"

# GIT_COMMIT=$(git log -1 --pretty=%h)
# BUILD_TIMESTAMP=$(date '+%F_%H:%M:%S')

docker buildx build . \
    -t "$ORGNIZATION/$NAME:$TAG"
    # --load --progress=plain \
