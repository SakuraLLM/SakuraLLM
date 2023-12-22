import time
import asyncio
from pathlib import Path
from threading import Lock
# from asyncio import Lock
from dataclasses import dataclass
from pprint import pformat, pprint

from pydantic import BaseModel

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sampler_hijack import hijack_samplers

from typing import *

import utils
from utils import consts, log_generation_config

import logging

if TYPE_CHECKING:
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from llama_cpp import Llama
    from auto_gptq import AutoGPTQForCausalLM
else:
    # FIXME(kuriko): try to making linting system happy
    Llama = AutoGPTQForCausalLM = LlamaForCausalLM = Any

ModelTypes = AutoGPTQForCausalLM | Llama | LlamaForCausalLM | AutoModelForCausalLM


logger = logging.getLogger(__name__)

@dataclass
class SakuraModelConfig:
    # read from console
    model_name_or_path: str
    use_gptq_model: bool
    trust_remote_code: bool = False
    text_length: int = 512

    # llama.cpp
    llama: bool = False
    llama_cpp: bool = False
    use_gpu: bool = False
    n_gpu_layers: int = 0

    # read from config.json (model_name_or_path)
    model_name: str|None = None
    model_quant: str|None = None
    model_version: str = "0.8"  # Can be also read from terminal, double check this.


def load_model(args: SakuraModelConfig):
    # args checker
    if args.llama_cpp and args.use_gptq_model:
        raise ValueError("You are using both use_gptq_model and llama_cpp flag, which is not supported.")

    if not args.llama_cpp and (args.use_gpu or args.n_gpu_layers != 0):
        logger.warning("You are using both use_gpu and n_gpu_layers flag without --llama_cpp.")
        if args.trust_remote_code is False and args.model_version in "0.5 0.7 0.8 0.9":
            raise ValueError("If you use model version 0.5, 0.7, 0.8 or 0.9, please add flag --trust_remote_code.")

    if args.llama:
        from transformers import LlamaForCausalLM, LlamaTokenizer

    if args.llama_cpp:
        from llama_cpp import Llama

    if args.use_gptq_model:
        from auto_gptq import AutoGPTQForCausalLM

    logger.info("loading model ...")

    if not args.llama_cpp:
        if args.llama:
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=args.trust_remote_code, use_safetensors=False)
    else:
        tokenizer = None

    if args.use_gptq_model:
        model = AutoGPTQForCausalLM.from_quantized(
            args.model_name_or_path,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            use_safetensors=False,
            use_triton=False,
            low_cpu_mem_usage=True,
        )
    elif args.llama:
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=args.trust_remote_code)
    elif args.llama_cpp:
        if args.use_gpu:
            n_gpu = -1 if args.n_gpu_layers == 0 else args.n_gpu_layers
            offload_kqv = True
        else:
            n_gpu = 0
            offload_kqv = False
        model = Llama(model_path=args.model_name_or_path, n_gpu_layers=n_gpu, n_ctx=4 * args.text_length, offload_kqv=offload_kqv)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=args.trust_remote_code, use_safetensors=False)

    return (tokenizer, model)

def get_llama_cpp_metadata(args: SakuraModelConfig):
    file = Path(args.model_name_or_path)
    filename = file.stem
    metadata = filename.split("-")
    model_version, model_quant = metadata[-2], metadata[-1]
    model_name = "-".join(metadata[:-2])
    return model_name, model_version, model_quant

class SakuraModel:
    # typing
    class ModelResponse(BaseModel):
        prompt_token: int
        new_token: int
        text: str
        finish_reason: str


    def __init__(
        self,
        cfg: SakuraModelConfig,
    ):
        self.cfg = cfg

        hijack_samplers()

        # Global lock for model generation.
        self.lock = Lock()

        (tokenizer, model) = load_model(cfg)
        self.tokenizer = tokenizer
        self.model = model

        try:
            if not cfg.llama_cpp:
                model_name = self.model.config.sakura_name
                model_version = self.model.config.sakura_version
                model_quant = self.model.config.sakura_quant

                # FIXME(kuriko): Currently we only know quant version from terminal, we cannot depend on parsing `model_name_or_path`
                # FIXME(kuriko): sakura_xxx is hard coded here, maybe we can find a better way.
                if (input_model_version := self.cfg.model_version) != model_version:
                    logger.error(f"Model version check failed, {input_model_version} != {model_version}")
                    logger.error(f"Current config: {model_name}-v{model_version}-{model_quant}")
                    exit(-1)

            else:
                # FIXME(kuriko): for llama_cpp model, we cannot decide so hard coded here.
                model_name, model_version, model_quant = get_llama_cpp_metadata(self.cfg)

            self.cfg.model_name = model_name
            self.cfg.model_version = model_version
            self.cfg.model_quant = model_quant
        except:
            logger.error("Some attrs are missing for this model {name|version|quant}, please check the model config!")
            exit(-1)

        return

    def get_cfg(self) -> SakuraModelConfig:
        return self.cfg

    def check_model_by_magic(self) -> bool:
        (prompt, ground_truth, output) = self.test_loaded()
        logger.debug(f"test output: {output}")
        logger.debug(f"ground truth: {ground_truth}")
        ret = ground_truth == output.text
        if not ret:
            logging.warning(f"model output is not correct, please check the loaded model")
            logging.warning(f"input: {prompt}")
            logging.warning(f"ground_truth: {ground_truth}")
            logging.warning(f"current output: {output}")
        return ret

    def make_prompt(self, system, user):
        if '0.9' in self.cfg.model_version:
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        elif '0.8' in self.cfg.model_version:
            prompt = f"<reserved_106>{user}<reserved_107>"
        else:
            raise ValueError(f"Wrong model version{self.cfg.model_version}, please view https://github.com/SakuraLLM/Sakura-13B-Galgame")
        return prompt

    def make_prompts_unstable(self, messages: list[dict[str, str]]) -> str:
        prompt = ""
        logger.debug(f"messages input is {str(messages)}")
        if messages[0]['role'] == 'user' and len(messages) == 1:
            prompt = self.make_prompt("", messages[0]['content'])
            src_text = messages[0]['content'].replace("将下面的日文文本翻译成中文：", "")
        else:
            if len(messages) == 2:
                if messages[0]['role'] == 'system' and messages[1]['role'] == 'user':
                    prompt = self.make_prompt(messages[0]['content'], messages[1]['content'])
                    src_text = messages[1]['content'].replace("将下面的日文文本翻译成中文：", "")
                else:
                    raise ValueError(f"Wrong messages format: {str(messages)}")
            else:
                raise ValueError(f"Wrong messages format: {str(messages)}")
        return prompt, src_text

    def get_max_text_length(self, length: int) -> int:
        return max(self.cfg.text_length, length)

    def completion(self, prompt: str, generation_config: GenerationConfig, is_print_speed: bool = True) -> ModelResponse:
        log_generation_config(generation_config)

        output = self.get_model_response(
            self.model,
            self.tokenizer,
            prompt,
            self.cfg.model_version,
            generation_config,
            self.get_max_text_length(len(prompt)),
            is_print_speed)

        return output

    async def completion_async(self, prompt: str, generation_config: GenerationConfig, is_print_speed: bool = True) -> ModelResponse:
        log_generation_config(generation_config)

        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(
            None,
            lambda: self.completion(
                prompt=prompt,
                generation_config=generation_config,
                is_print_speed=is_print_speed,
            )
        )

        return output

    def test_loaded(self) -> (str, ModelResponse):
        testcase = consts.get_test_case_by_model_version(self.cfg.model_name, self.cfg.model_version, self.cfg.model_quant)

        generation_config = testcase.generation_config
        test_input = testcase.test_input
        test_output = testcase.test_output

        prompt = consts.get_prompt(test_input, self.cfg.model_name, self.cfg.model_version, self.cfg.model_quant)

        output = self.completion(prompt, generation_config, is_print_speed=False)

        return prompt, test_output, output

    def __llama_cpp_model(self, model: "Llama", prompt: str, generation_config: GenerationConfig):
        output = model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'], temperature=generation_config.__dict__['temperature'], top_p=generation_config.__dict__['top_p'], repeat_penalty=generation_config.__dict__['repetition_penalty'])
        response = output['choices'][0]['text']
        pprint(output)

        input_tokens_len = output['usage']['prompt_tokens']
        new_tokens = output['usage']['completion_tokens']
        return response, (input_tokens_len, new_tokens)


    def __general_model(self, model: ModelTypes, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig):
        input_tokens = tokenizer(prompt, return_tensors="pt")
        input_tokens_len = input_tokens.input_ids.shape[-1]

        generation = model.generate(**input_tokens.to(model.device), generation_config=generation_config)[0]

        new_tokens = generation.shape[-1] - input_tokens_len

        response = tokenizer.decode(generation)
        output = utils.split_response(response, model_version)
        return output, (input_tokens_len, new_tokens)


    def get_model_response(self, model: ModelTypes, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig, text_length: int, is_print_speed:bool=True) -> ModelResponse:
        for i in range(3):
            with self.lock:  # using lock to prevent too many memory allocated on GPU
                t0 = time.time()
                if self.cfg.llama_cpp:
                    output, (input_tokens_len, new_tokens) = self.__llama_cpp_model(model, prompt, generation_config)
                else:
                    output, (input_tokens_len, new_tokens) = self.__general_model(model, tokenizer, prompt, model_version, generation_config)
                t1 = time.time()

            # FIXME(kuriko): a temporary solution to avoid empty output in llama.cpp
            if len(output) == 0:
                generation_config.__dict__['temperature'] = 1.0
                generation_config.__dict__['top_p'] = 1.0
                logger.error(f"Model output is empty, retrying ({i}/3)..., This is a very rare situation, please report to devs")
                continue

            if is_print_speed:
                logger.info(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {input_tokens_len} tokens)')

            # whether model stops because eos token or length limit
            if new_tokens == generation_config.__dict__['max_new_tokens']:
                finish_reason = "length"
            else:
                finish_reason = "stop"

            return self.ModelResponse(
                prompt_token = input_tokens_len,
                new_token = new_tokens,
                text = output,
                finish_reason = finish_reason
            )

        return self.ModelResponse(
                prompt_token = input_tokens_len,
                new_token = new_tokens,
                text = "模型生成出错，这是一个非常罕见的问题。原文为：" + generation_config.__dict__['src_text'],
                finish_reason = "stop"
            )

    def get_model_response_anti_degen(self, model: ModelTypes, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig, text_length: int):
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

        t0 = time.time()
        if self.cfg.llama_cpp:
            output, (input_tokens_len, new_tokens) = self.__llama_cpp_model(model, prompt, generation_config)
        else:
            output, (input_tokens_len, new_tokens) = self.__general_model(model, tokenizer, prompt, model_version, generation_config)
        t1 = time.time()
        stage = 0
        while new_tokens == generation_config.__dict__['max_new_tokens']:
            stage += 1
            if stage > 2:
                print("model degeneration cannot be avoided.")
                break
            if self.cfg.llama_cpp:
                output, (input_tokens_len, new_tokens) = self.__llama_cpp_model(model, prompt, generation_config)
            else:
                output, (input_tokens_len, new_tokens) = self.__general_model(model, tokenizer, prompt, model_version, generation_config)
        return output
