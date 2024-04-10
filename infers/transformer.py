'''
WIP:
'''
import logging
from typing import *

from transformers import GenerationConfig, AutoTokenizer

import utils
from infers import BaseInferEngine
from utils.model import SakuraModelConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from auto_gptq import AutoGPTQForCausalLM

    ModelTypes = LlamaForCausalLM | AutoGPTQForCausalLM
    TokenizerTypes = AutoTokenizer | LlamaTokenizer
else:
    ModelTypes = TokenizerTypes = Any


class TransformerEngine(BaseInferEngine):
    def __init__(self, model: ModelTypes, tokenizer: TokenizerTypes):
        self.tokenizer = tokenizer
        self.model = model

        _, _, model_version = self.get_metadata(None);
        self.model_version = model_version
        return

    def get_metadata(self, _: SakuraModelConfig):
        model_name = self.model.config.sakura_name
        model_version = self.model.config.sakura_version
        model_quant = self.model.config.sakura_quant
        return model_name, model_quant, model_version

    def generate(self, prompt: str, generation_config: GenerationConfig):
        input_tokens = self.tokenizer(prompt, return_tensors="pt")
        input_tokens_len = input_tokens.input_ids.shape[-1]

        generation = self.model.generate(**input_tokens.to(self.model.device), generation_config=generation_config)[0]

        new_tokens = generation.shape[-1] - input_tokens_len

        response = self.tokenizer.decode(generation)
        output = utils.split_response(response, self.model_version)
        return output, (input_tokens_len, new_tokens)

    def stream_generate(self, messages: list[dict[str, str]], generation_config: GenerationConfig):
        def parse_messages():
            if messages[0]['role'] == "system":
                _system = messages.pop(0)['content']
            else:
                _system = ""
            _query = messages.pop(-1)['content']
            return _system, messages, _query

        position = 0
        start = 0
        token_cnt = 0
        user = messages[-1]
        system, history, query = parse_messages()
        print(user)
        if "0.8" in self.model_version:
            generation_config.user_token_id = 195
            generation_config.assistant_token_id = 196
            generation_config.pad_token_id = 0
            generation_config.bos_token_id = 1
            generation_config.eos_token_id = 2
            self.model.generation_config.__dict__ = generation_config.__dict__
            for response in self.model.chat(self.tokenizer, history + [user], stream=True,
                                            generation_config=generation_config):
                token_cnt += 1
                position = len(response)
                yield response[start:position], None
                start = position
        elif "0.9" in self.model_version:
            generation_config.chat_format = 'chatml'
            generation_config.max_window_size = 6144
            generation_config.pad_token_id = 151643
            generation_config.eos_token_id = 151643
            self.model.generation_config.__dict__ = generation_config.__dict__
            for response in self.model.chat_stream(self.tokenizer, query, history=history, system=system,
                                                   generation_config=generation_config):
                token_cnt += 1
                position = len(response)
                yield response[start:position], None
                start = position
        if token_cnt == generation_config.__dict__['max_new_tokens']:
            yield "", "length"
        else:
            yield "", "stop"
