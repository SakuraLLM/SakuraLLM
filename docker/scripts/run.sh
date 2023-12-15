#!/bin/bash

source docker/scripts/env.sh

if [[ $COMMAND == "translate_epub" ]]; then
    python3 translate_epub.py \
            --use_gptq_model --trust_remote_code --text_length 512 \
            --model_name_or_path "${SAKURA_MODEL_PATH}" \
            --model_version "${SAKURA_MODEL_VERSION}" \
            $@
elif [[ $COMMAND == "translate_novel" ]]; then
    python3 translate_novel.py \
            --use_gptq_model --trust_remote_code --text_length 512 \
            --model_name_or_path "${SAKURA_MODEL_PATH}" \
            --model_version "${SAKURA_MODEL_VERSION}" \
            $@
elif [[ $COMMAND == "run_server" ]]; then
    python3 server.py \
            --use_gptq_model --trust_remote_code \
            --listen "${ADDRESS}:${PORT}" \
            --model_name_or_path "${SAKURA_MODEL_PATH}" \
            --model_version "${SAKURA_MODEL_VERSION}" \
            --auth "${USERNAME}:${PASSWORD}" \
            --log "${LOGLEVEL}"
else
    echo "Please provide command: run_server, translate_epub, translate_novel"
fi
