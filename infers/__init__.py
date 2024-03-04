from typing import *
from transformers import GenerationConfig
from utils.model import SakuraModelConfig

class BaseInferEngine:
    def get_metadata(self, args: SakuraModelConfig):
        raise NotImplemented

    def generate(self, prompt: str, generation_config: GenerationConfig):
        raise NotImplemented

    def stream_generate(self, prompt: str, generation_config: GenerationConfig):
        raise NotImplemented
