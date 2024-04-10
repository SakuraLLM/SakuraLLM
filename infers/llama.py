import logging
from pathlib import Path
from pprint import pprint

from llama_cpp import Llama
from transformers import GenerationConfig

from infers import BaseInferEngine
from utils.model import SakuraModelConfig

logger = logging.getLogger(__name__)


class LlamaCpp(BaseInferEngine):
    def __init__(self, args: SakuraModelConfig):
        if args.use_gpu:
            n_gpu = -1 if args.n_gpu_layers == 0 else args.n_gpu_layers
            offload_kqv = True
        else:
            n_gpu = 0
            offload_kqv = False

        self.model = Llama(
            model_path=args.model_name_or_path,
            n_gpu_layers=n_gpu,
            n_ctx=4 * args.text_length,
            offload_kqv=offload_kqv,
        )
        return

    def get_metadata(self, args: SakuraModelConfig):
        file = Path(args.model_name_or_path)
        filename = file.stem
        metadata = filename.split("-")
        model_version, model_quant = metadata[-2], metadata[-1]
        model_name = "-".join(metadata[:-2])
        return model_name, model_version, model_quant

    def generate(self, prompt: str, generation_config: GenerationConfig):
        output = self.model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'],
                            temperature=generation_config.__dict__['temperature'],
                            top_p=generation_config.__dict__['top_p'],
                            repeat_penalty=generation_config.__dict__['repetition_penalty'],
                            frequency_penalty=generation_config.__dict__['frequency_penalty'])
        response = output['choices'][0]['text']
        pprint(output)

        input_tokens_len = output['usage']['prompt_tokens']
        new_tokens = output['usage']['completion_tokens']
        return response, (input_tokens_len, new_tokens)

    def stream_generate(self, prompt: str, generation_config: GenerationConfig):
        logger.debug(f"prompt is: {prompt}")
        for output in self.model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'],
                                 temperature=generation_config.__dict__['temperature'],
                                 top_p=generation_config.__dict__['top_p'],
                                 repeat_penalty=generation_config.__dict__['repetition_penalty'],
                                 frequency_penalty=generation_config.__dict__['frequency_penalty'], stream=True):
            yield output['choices'][0]['text'], output['choices'][0]['finish_reason']
