#!/bin/bash

if [ ! -e "${SAKURA_MODEL_PATH}/config.json" ]; then
    tree "${SAKURA_MODEL_PATH}/config.json"
    echo "Please use -v<path to models>:/models to bind model"
    echo "and make sure that /models/${SAKURA_MODEL_PATH} exists"
    exit -1
fi