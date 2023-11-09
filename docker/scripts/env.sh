#!/bin/bash

if [ ! -e "${SAKURA_MODEL_PATH}/config.json" ]; then
    tree "${SAKURA_MODEL_PATH}/config.json"
    echo "Please use -v<MODEL_PATH>:/model/$MODEL_NAME to bind model"
    exit -1
fi

cp ${MODEL_ROOT}/tokenization_baichuan.py ${SAKURA_MODEL_PATH}/