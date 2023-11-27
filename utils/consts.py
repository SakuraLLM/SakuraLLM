from dataclasses import dataclass
from pydantic import BaseModel
from transformers import GenerationConfig
from types import *
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelTestCase:
    model_version: str
    generation_config: GenerationConfig
    test_input: str
    test_output: str


def get_prompt(input, model_name:str, model_version:str, model_quant:str=None):
    # FIXME(kuriko): hardcoded here for llama_cpp quant model
    if model_name == 'llama_cpp':
        prompt = "<reserved_106>将下面的日文文本翻译成中文：" + input + "<reserved_107>"
        return prompt

    if model_version == '0.5' or model_version == '0.8':
        prompt = "<reserved_106>将下面的日文文本翻译成中文：" + input + "<reserved_107>"
        return prompt
    if model_version == '0.7':
        prompt = f"<|im_start|>user\n将下面的日文文本翻译成中文：{input}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    if model_version == '0.1':
        prompt = "Human: \n将下面的日文文本翻译成中文：" + input + "\n\nAssistant: \n"
        return prompt
    if model_version == '0.4':
        prompt = "User: 将下面的日文文本翻译成中文：" + input + "\nAssistant: "
        return prompt

    raise ValueError(f"Wrong model version{model_version}, please view https://huggingface.co/sakuraumi/Sakura-13B-Galgame")


def get_test_case_by_model_version(model_name:str, model_version:str, model_quant:str):
    default_generation_config = GenerationConfig(
        num_beams=1,
        max_new_tokens=1024,
        min_new_tokens=1,
        do_sample=False,
    )

    if model_name == "llama_cpp" or model_version == "0.8":
        return ModelTestCase(
            model_version=model_version,
            generation_config=default_generation_config,
            test_input="やる気マンゴスキン",
            test_output="干劲Mangoskin",
        )
    else:
        logger.error(f"Invalid model version: {model_version}, please check the arguments")
        return None
