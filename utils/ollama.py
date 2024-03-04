import multiprocessing
import subprocess
import ollama
import time
from tqdm import tqdm


class Ollama:
    '''Llama-style wrapper for ollama'''
    def __init__(self, model):
        self.model = model
        self.start()
        time.sleep(5)  # wait for ollama serve to start
        self.pull()

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


def ollama_serve():
    try:
        p = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE)
        for line in p.stdout:
            print(line.decode(), end='')
    except FileNotFoundError:
        raise FileNotFoundError("ollama app not found. Have you installed it?")
