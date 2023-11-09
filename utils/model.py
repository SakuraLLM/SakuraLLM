from functools import lru_cache
import re
import time
import random
from pathlib import Path
from threading import Thread, Lock

from pydantic import BaseModel

from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sampler_hijack import hijack_samplers

from typing import *

import utils
from utils import consts

import logging

logger = logging.getLogger(__name__)

class SakuraModelConfig:
    model_name_or_path: str
    use_gptq_model: bool
    model_version: str = "0.8"
    trust_remote_code: bool = False
    llama: bool = False
    text_length: int = 512

def load_model(args: SakuraModelConfig):
    if args.use_gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        
    if args.llama:
        from transformers import LlamaForCausalLM, LlamaTokenizer
        
    if args.trust_remote_code is False and args.model_version in "0.5 0.7 0.8":
        raise ValueError("If you use model version 0.5, 0.7 or 0.8, please add flag --trust_remote_code.")

    logger.info("loading model ...")

    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=args.trust_remote_code, use_safetensors=False)

    if args.use_gptq_model:
        model = AutoGPTQForCausalLM.from_quantized(args.model_name_or_path, device="cuda:0", trust_remote_code=args.trust_remote_code, use_safetensors=False)
    else:
        if args.llama:
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=args.trust_remote_code)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=args.trust_remote_code, use_safetensors=False)

    return (tokenizer, model)


class SakuraModel:
    # typing
    class ModelResponse(BaseModel):
        new_token: int
        text: str


    def __init__(
        self,
        cfg: SakuraModelConfig,
    ):
        self.cfg = cfg

        hijack_samplers()

        (tokenizer, model) = load_model(cfg)
        self.tokenizer = tokenizer
        self.model = model

        self.lock = Lock()

        return
    
    def check_model_by_magic(self) -> bool:
        (ground_truth, output) = self.test_loaded()
        ret = ground_truth == output.text
        if not ret:
            logging.warning(f"model output is not correct, please check the loaded model")
            logging.warning(f"input: {consts.get_test_input(self.cfg.model_version)}")
            logging.warning(f"ground_truth: {ground_truth}")
            logging.warning(f"current output: {output}")
        return ret


    def get_max_text_length(self, length: int) -> int:
        return max(self.cfg.text_length, length)

    def completion(self, prompt: str, generation_config: GenerationConfig = utils.get_default_generation_config()) -> ModelResponse:
        output = self.get_model_response(
            self.model, 
            self.tokenizer, 
            prompt, 
            self.cfg.model_version, 
            generation_config, 
            self.get_max_text_length(len(prompt))
            )

        return output

    def test_loaded(self) -> (str, ModelResponse):
        test_input = consts.get_test_input(self.cfg.model_version)
        prompt = utils.get_prompt(test_input, self.cfg.model_version)
        output = self.completion(prompt, utils.get_default_generation_config())
        return consts.get_test_output(self.cfg.model_version), output


    def get_model_response(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig, text_length: int) -> ModelResponse:
        input_token = tokenizer(prompt, return_tensors="pt")
        input_token_len = input_token.input_ids.shape[-1]
        with self.lock:  # using lock to prevent to many memory allocated on GPU
            generation = model.generate(**input_token.to(model.device), generation_config=generation_config)[0]
        new_token = generation.shape[-1] - input_token_len
        response = tokenizer.decode(generation)
        output = utils.split_response(response, model_version)

        return self.ModelResponse(
            new_token = new_token,
            text = output, 
        )

    def get_model_response_anti_degen(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig, text_length: int):
        backup_generation_config_stage2 = GenerationConfig(
                temperature=0.1,
                top_p=0.3,
                top_k=40,
                num_beams=1,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                max_new_tokens=2 * text_length,
                min_new_tokens=1,
                do_sample=True,
                repetition_penalty=1.0,
                frequency_penalty=0.05
            )
        
        backup_generation_config_stage3 = GenerationConfig(
                temperature=0.1,
                top_p=0.3,
                top_k=40,
                num_beams=1,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                max_new_tokens=2 * text_length,
                min_new_tokens=1,
                do_sample=True,
                repetition_penalty=1.0,
                frequency_penalty=0.2
            )

        backup_generation_config = [backup_generation_config_stage2, backup_generation_config_stage3]

        generation = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), generation_config=generation_config)[0]
        if len(generation) > 2 * text_length:
            stage = 0
            while utils.detect_degeneration(list(generation), model_version):
                stage += 1
                if stage > 2:
                    print("model degeneration cannot be avoided.")
                    break
                generation = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), generation_config=backup_generation_config[stage-1])[0]
        response = tokenizer.decode(generation)
        output = utils.split_response(response, model_version)
        return output
