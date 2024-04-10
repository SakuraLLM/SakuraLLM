import logging
import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from pprint import pprint

import ollama
from tqdm import tqdm
from transformers import GenerationConfig

from infers import BaseInferEngine
from utils.model import SakuraModelConfig

logger = logging.getLogger(__name__)


class Ollama(BaseInferEngine):
    '''Llama-style wrapper for ollama'''

    def __init__(self, args: SakuraModelConfig):
        self.model = args.model_name_or_path
        if self.check_ollama():
            self.start()
            time.sleep(5)
            self.pull()
        else:
            logger.error("ollama executable not found in path. Have you installed it?")
            exit(-1)

    def __call__(self, prompt, stream=False, **kwargs):
        return ollama.generate(self.model, prompt, stream=stream, options=kwargs)

    def pull(self):
        current_digest, bars = "", {}
        for progress in ollama.pull(self.model, stream=True):
            digest = progress.get("digest", "")
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()
            if not digest:
                print(progress.get("status"))
                continue
            if digest not in bars and (total := progress.get("total")):
                bars[digest] = tqdm(
                    total=total,
                    desc=f"pulling {digest[7:19]}",
                    unit="B",
                    unit_scale=True,
                )
            if completed := progress.get("completed"):
                bars[digest].update(completed - bars[digest].n)
            current_digest = digest

    def start(self):
        proc = multiprocessing.Process(target=ollama_serve)
        proc.start()

    def check_ollama(self):
        env_paths = os.environ["PATH"]
        if os.name == "nt":
            ollama_bin = "ollama.exe"
            env_paths = env_paths.split(";")
        elif os.name == "posix":
            ollama_bin = "ollama"
            env_paths = env_paths.split(":")
        for path in env_paths:
            ollama_path = Path(path).joinpath(ollama_bin)
            if ollama_path.exists():
                return True
        return False

    def get_metadata(self, args: SakuraModelConfig):
        metadata = args.model_name_or_path.split("-")
        model_version, model_quant = metadata[-2], metadata[-1]
        model_name = "-".join(metadata[:-2])
        return model_name, model_version, model_quant

    def generate(self, prompt: str, generation_config: GenerationConfig):
        output = self(
            prompt,
            num_ctx=generation_config.__dict__["max_new_tokens"],
            temperature=generation_config.__dict__["temperature"],
            top_p=generation_config.__dict__["top_p"],
            repeat_penalty=generation_config.__dict__["repetition_penalty"],
            frequency_penalty=generation_config.__dict__["frequency_penalty"],
        )
        response = output["response"]
        pprint(output)

        # FIXME(Isotr0py): According to the #2068 issue of ollama (https://github.com/ollama/ollama/issues/2068),
        # prompt_eval_count may disappear.
        if "prompt_eval_count" not in output:
            # NOTE(kuriko): Under most cases, input_tokens_len is not used.
            input_tokens_len = None
        else:
            input_tokens_len = output["prompt_eval_count"]
        new_tokens = output['eval_count']
        return response, (input_tokens_len, new_tokens)

    def stream_generate(self, prompt: str, generation_config: GenerationConfig):
        logger.debug(f"prompt is: {prompt}")
        for output in self(
                prompt,
                stream=True,
                num_ctx=generation_config.__dict__["max_new_tokens"],
                temperature=generation_config.__dict__["temperature"],
                top_p=generation_config.__dict__["top_p"],
                repeat_penalty=generation_config.__dict__["repetition_penalty"],
                frequency_penalty=generation_config.__dict__["frequency_penalty"],
        ):
            finish_reason = "stop" if output['done'] else None
            yield output["response"], finish_reason


def ollama_serve():
    p = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE)
    for line in p.stdout:
        print(line.decode(), end='')
