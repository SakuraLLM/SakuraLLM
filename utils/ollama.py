import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import ollama
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


class Ollama:
    '''Llama-style wrapper for ollama'''
    def __init__(self, model):
        self.model = model
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


def ollama_serve():
    p = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE)
    for line in p.stdout:
        print(line.decode(), end='')
