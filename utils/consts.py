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


def get_prompt(raw_jp_text, model_name:str, model_version:str, model_quant:str=None, gpt_dict:list=[]):
    # FIXME(kuriko): hardcoded here for llama_cpp quant model
    if model_name == 'llama_cpp':
        prompt = "<reserved_106>将下面的日文文本翻译成中文：" + raw_jp_text + "<reserved_107>"
        return prompt

    if model_version == '0.5' or "0.8" in model_version:
        prompt = "<reserved_106>将下面的日文文本翻译成中文：" + raw_jp_text + "<reserved_107>"
        return prompt
    if "0.9" in model_version:
        prompt = f"<|im_start|>system\n你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。<|im_end|>\n<|im_start|>user\n将下面的日文文本翻译成中文：{raw_jp_text}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    if "0.10" in model_version:
        gpt_dict_text_list = []
        for gpt in gpt_dict:
            src = gpt['src']
            dst = gpt['dst']
            info = gpt['info'] if "info" in gpt.keys() else None
            if info:
                single = f"{src}->{dst} #{info}"
            else:
                single = f"{src}->{dst}"
            gpt_dict_text_list.append(single)

        gpt_dict_raw_text = "\n".join(gpt_dict_text_list)

        user_prompt = "根据以下术语表（可以为空）：\n" + gpt_dict_raw_text + "\n\n" + "将下面的日文文本根据上述术语表的对应关系和备注翻译成中文：" + raw_jp_text
        prompt = "<|im_start|>system\n你是一个轻小说翻译模型，可以流畅通顺地使用给定的术语表以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，不要擅自添加原文中没有的代词，也不要擅自增加或减少换行。<|im_end|>\n" \
        + "<|im_start|>user\n" + user_prompt + "<|im_end|>\n" \
        + "<|im_start|>assistant\n" # assistant prompt start
        return prompt
    if model_version == '0.7':
        prompt = f"<|im_start|>user\n将下面的日文文本翻译成中文：{raw_jp_text}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    if model_version == '0.1':
        prompt = "Human: \n将下面的日文文本翻译成中文：" + raw_jp_text + "\n\nAssistant: \n"
        return prompt
    if model_version == '0.4':
        prompt = "User: 将下面的日文文本翻译成中文：" + raw_jp_text + "\nAssistant: "
        return prompt

    raise ValueError(f"Wrong model version{model_version}, please view https://huggingface.co/sakuraumi/Sakura-13B-Galgame")


def get_test_case_by_model_version(model_name:str, model_version:str, model_quant:str):
    default_generation_config = GenerationConfig(
        num_beams=1,
        max_new_tokens=20,
        min_new_tokens=1,
        do_sample=False,
    )

    if model_name == "llama_cpp" or model_version in "0.8 0.9 0.10":
        return ModelTestCase(
            model_version=model_version,
            generation_config=default_generation_config,
            test_input="やる気マンゴスキン",
            test_output="干劲Mangoskin",
        )
    else:
        logger.error(f"Invalid model version: {model_version}, please check the arguments")
        return None
