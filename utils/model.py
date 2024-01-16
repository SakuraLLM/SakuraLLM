import time
import utils
import logging
import copy
import traceback
from pathlib import Path
from threading import Lock
# from asyncio import Lock
from dataclasses import dataclass
from pprint import pformat, pprint
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PretrainedConfig
from sampler_hijack import hijack_samplers
from typing import *
from utils import consts, log_generation_config

if TYPE_CHECKING:
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from llama_cpp import Llama
    from auto_gptq import AutoGPTQForCausalLM
    from vllm import AsyncLLMEngine, LLM
else:
    # FIXME(kuriko): try to making linting system happy
    Llama = AutoGPTQForCausalLM = LlamaForCausalLM = AsyncLLMEngine = Any

ModelTypes = AutoGPTQForCausalLM | Llama | LlamaForCausalLM | AutoModelForCausalLM | AsyncLLMEngine


logger = logging.getLogger(__name__)

@dataclass
class SakuraModelConfig:
    # read from console
    model_name_or_path: str
    use_gptq_model: bool
    use_awq_model: bool
    trust_remote_code: bool = False
    text_length: int = 512

    # llama.cpp
    llama: bool = False
    llama_cpp: bool = False
    use_gpu: bool = False
    n_gpu_layers: int = 0

    # vllm
    vllm: bool = False
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

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

    if args.vllm:
        from vllm import AsyncEngineArgs, AsyncLLMEngine, LLM
        from vllm.utils import Counter

        class MixLLMEngine(LLM):
            "an AsyncLLMEngine unwrapper for flexible generation"
            def __init__(self, llm_engine: AsyncLLMEngine):
                self.llm_engine = llm_engine.engine
                self.async_engine = llm_engine
                self.request_counter = Counter()
            # TODO(Isotr0py): implement llama.cpp-like generate method
            # def generate(self, sampling_params, stream=False):
            #     raise NotImplementedError("Not implemented yet.")

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

    if args.use_gptq_model and not args.vllm:
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
    elif args.vllm:
        if args.use_gptq_model:
            quantization = "gptq"
        elif args.use_awq_model:
            quantization = "awq"
        else:
            quantization = None
        engine_args = AsyncEngineArgs(
            model=args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            quantization=quantization,
            enforce_eager=args.enforce_eager,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        model = MixLLMEngine(engine)
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
                if cfg.vllm:
                    # vllm Engine doesn't have config attr, we need to reload config from pretrained
                    config = PretrainedConfig.from_pretrained(self.cfg.model_name_or_path)
                else:
                    config = self.model.config
                model_name = config.sakura_name
                model_version = config.sakura_version
                model_quant = config.sakura_quant

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

    def make_continue_prompt(self, role, value):
        if '0.9' in self.cfg.model_version:
            prompt = f"<|im_start|>{role}\n{value}<|im_end|>\n"
        elif '0.8' in self.cfg.model_version:
            try:
                str_map = {"user": "<reserved_106>", "assistant": "<reserved_107>"}
                prompt = f"{str_map[role]}{value}"
            except KeyError as e:
                prompt = ""
                logger.warning(f"Role '{role}' of this model is not supported, ignored.")
        else:
            raise ValueError(f"Wrong model version{self.cfg.model_version}, please view https://github.com/SakuraLLM/Sakura-13B-Galgame")
        return prompt

    def make_end_prompt(self):
        role = 'assistant'
        if '0.9' in self.cfg.model_version:
            prompt = f"<|im_start|>{role}\n"
        elif '0.8' in self.cfg.model_version:
            str_map = {"user": "<reserved_106>", "assistant": "<reserved_107>"}
            prompt = f"{str_map[role]}"
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

    def check_messages(self, messages: list[dict[str, str]]) -> None:
        _messages = copy.deepcopy(messages)
        if _messages[-1]['role'] != 'user':
            raise ValueError(f"Wrong messages format: {str(messages)}")
        if _messages[0]['role'] == 'system':
            _messages.pop(0)
        for message in _messages:
            if message['role'] == 'system':
                logger.warning(f"Wrong messages format detected: {str(messages)}")
        if len(_messages) % 2 != 1:
            logger.warning(f"Wrong messages format detected: {str(messages)}")

    def make_prompt_stable(self, messages: list[dict[str, str]]) -> str:
        prompt = ""
        logger.debug(f"MAKE_PROMPT_STABLE: input is {str(messages)}")
        self.check_messages(messages)
        for idx, message in enumerate(messages):
            prompt += self.make_continue_prompt(message['role'], message['content'])
        prompt += self.make_end_prompt()
        logger.debug(f"MAKE_PROMPT_STABLE: prompt is {prompt}")
        return prompt

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

    def completion_stream(self, messages: list[dict[str, str]], generation_config: GenerationConfig, is_print_speed: bool = True) -> ModelResponse:
        log_generation_config(generation_config)

        for output, finish_reason in self.get_model_response_stream(
            self.model,
            self.tokenizer,
            messages,
            self.cfg.model_version,
            generation_config,
            is_print_speed):

            yield output, finish_reason

    def test_loaded(self) -> (str, ModelResponse):
        testcase = consts.get_test_case_by_model_version(self.cfg.model_name, self.cfg.model_version, self.cfg.model_quant)

        generation_config = testcase.generation_config
        test_input = testcase.test_input
        test_output = testcase.test_output

        prompt = consts.get_prompt(test_input, self.cfg.model_name, self.cfg.model_version, self.cfg.model_quant)

        output = self.completion(prompt, generation_config, is_print_speed=False)

        return prompt, test_output, output

    def __llama_cpp_model(self, model: "Llama", prompt: str, generation_config: GenerationConfig):
        output = model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'], temperature=generation_config.__dict__['temperature'], top_p=generation_config.__dict__['top_p'], repeat_penalty=generation_config.__dict__['repetition_penalty'], frequency_penalty=generation_config.__dict__['frequency_penalty'])
        response = output['choices'][0]['text']
        pprint(output)

        input_tokens_len = output['usage']['prompt_tokens']
        new_tokens = output['usage']['completion_tokens']
        return response, (input_tokens_len, new_tokens)

    def __llama_cpp_model_stream(self, model: "Llama", prompt: str, generation_config: GenerationConfig):
        logger.debug(f"prompt is: {prompt}")
        for output in model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'], temperature=generation_config.__dict__['temperature'], top_p=generation_config.__dict__['top_p'], repeat_penalty=generation_config.__dict__['repetition_penalty'], frequency_penalty=generation_config.__dict__['frequency_penalty'], stream=True):
            yield output['choices'][0]['text'], output['choices'][0]['finish_reason']

    def __vllm_model(self, model: "LLM", prompt: str, generation_config: GenerationConfig):
        from vllm import SamplingParams

        logger.debug(f"prompt is: {prompt}")
        sampling_params = SamplingParams(
            max_tokens=generation_config.__dict__['max_new_tokens'],
            temperature=generation_config.__dict__['temperature'],
            top_p=generation_config.__dict__['top_p'],
            repetition_penalty=generation_config.__dict__['repetition_penalty'],
            frequency_penalty=generation_config.__dict__['frequency_penalty'],
        )
        # need to unwrap the async engine for generation
        output = model.generate(prompt, sampling_params)
        request_output = output[0].outputs[0]
        text = request_output.text
        input_tokens_len = len(output[0].prompt_token_ids)
        new_tokens = len(request_output.token_ids)
        return text, (input_tokens_len, new_tokens)

    def __vllm_model_stream(self, model: "AsyncLLMEngine", prompt: str, generation_config: GenerationConfig):
        import asyncio
        import nest_asyncio
        from vllm import SamplingParams

        # FIXME(Isotr0py): hard to code without nest_asyncio
        nest_asyncio.apply()

        logger.debug(f"prompt is: {prompt}")
        sampling_params = SamplingParams(
            max_tokens=generation_config.__dict__['max_new_tokens'],
            temperature=generation_config.__dict__['temperature'],
            top_p=generation_config.__dict__['top_p'],
            repetition_penalty=generation_config.__dict__['repetition_penalty'],
            frequency_penalty=generation_config.__dict__['frequency_penalty'],
        )
        generator = model.async_engine.generate(prompt, sampling_params, request_id="Sakura")
        # TODO(Isotr0py): replace with llama.cpp-like streaming generation method
        previous_output = ""
        while True:
            try:
                output = asyncio.run(anext(generator))
                output_text = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason
                delta_text = output_text.removeprefix(previous_output)
                previous_output = output_text
                yield delta_text, finish_reason
            except StopAsyncIteration:
                break

    def __general_model(self, model: ModelTypes, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig):
        input_tokens = tokenizer(prompt, return_tensors="pt")
        input_tokens_len = input_tokens.input_ids.shape[-1]

        generation = model.generate(**input_tokens.to(model.device), generation_config=generation_config)[0]

        new_tokens = generation.shape[-1] - input_tokens_len

        response = tokenizer.decode(generation)
        output = utils.split_response(response, model_version)
        return output, (input_tokens_len, new_tokens)

    def __general_model_stream(self, model: ModelTypes, tokenizer: AutoTokenizer, messages: list[dict[str, str]], model_version: str, generation_config: GenerationConfig):

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
        self.check_messages(messages)
        user = messages[-1]
        system, history, query = parse_messages()
        print(user)
        if "0.8" in self.cfg.model_version:
            generation_config.user_token_id = 195
            generation_config.assistant_token_id = 196
            generation_config.pad_token_id = 0
            generation_config.bos_token_id = 1
            generation_config.eos_token_id = 2
            model.generation_config.__dict__ = generation_config.__dict__
            for response in model.chat(tokenizer, history + [user], stream=True, generation_config=generation_config):
                token_cnt += 1
                position = len(response)
                yield response[start:position], None
                start = position
        elif "0.9" in self.cfg.model_version:
            generation_config.chat_format = 'chatml'
            generation_config.max_window_size = 6144
            generation_config.pad_token_id = 151643
            generation_config.eos_token_id = 151643
            model.generation_config.__dict__ = generation_config.__dict__
            for response in model.chat_stream(tokenizer, query, history=history, system=system, generation_config=generation_config):
                token_cnt += 1
                position = len(response)
                yield response[start:position], None
                start = position
        if token_cnt == generation_config.__dict__['max_new_tokens']:
            yield "", "length"
        else:
            yield "", "stop"

    def get_model_response(self, model: ModelTypes, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig, text_length: int, is_print_speed:bool=True) -> ModelResponse:
        for i in range(2):
            with self.lock:  # using lock to prevent too many memory allocated on GPU
                t0 = time.time()
                if self.cfg.llama_cpp:
                    output, (input_tokens_len, new_tokens) = self.__llama_cpp_model(model, prompt, generation_config)
                elif self.cfg.vllm:
                    output, (input_tokens_len, new_tokens) = self.__vllm_model(model, prompt, generation_config)
                else:
                    output, (input_tokens_len, new_tokens) = self.__general_model(model, tokenizer, prompt, model_version, generation_config)
                t1 = time.time()

            # FIXME(kuriko): a temporary solution to avoid empty output in llama.cpp
            if len(output) == 0 and i == 0:
                generation_config.__dict__['temperature'] = 1.0
                generation_config.__dict__['top_p'] = 1.0
                logger.warning(f"Model output is empty, retrying..., This is a very rare situation, please check your input or open an issue.")
                continue
            elif len(output) == 0:
                logger.warning(f"Model output is empty. This is a very rare situation, please check your input or open an issue.")

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

        # 理论上不会有触发这条代码的情况
        return self.ModelResponse(
            prompt_token = input_tokens_len,
            new_token = new_tokens,
            text = "",
            finish_reason = "stop"
        )

    def get_model_response_stream(self, model: ModelTypes, tokenizer: AutoTokenizer, messages: list[dict[str, str]], model_version: str, generation_config: GenerationConfig, is_print_speed:bool=True) -> ModelResponse:

        # with self.lock:  # using lock to prevent too many memory allocated on GPU
        t0 = time.time()
        token_cnt = 0
        if self.cfg.llama_cpp:
            prompt = self.make_prompt_stable(messages)
            for output, finish_reason in self.__llama_cpp_model_stream(model, prompt, generation_config):
                token_cnt += 1
                yield output, finish_reason
        elif self.cfg.vllm:
            prompt = self.make_prompt_stable(messages)
            for output, finish_reason in self.__vllm_model_stream(model, prompt, generation_config):
                token_cnt += 1
                yield output, finish_reason
        else:
            self.check_messages(messages)
            for output, finish_reason in self.__general_model_stream(model, tokenizer, messages, model_version, generation_config):
                token_cnt += 1
                yield output, finish_reason
        t1 = time.time()
        if is_print_speed:
                logger.info(f'Output generated in {(t1-t0):.2f} seconds ({token_cnt/(t1-t0):.2f} tokens/s, {token_cnt} tokens generated)')
        return

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
        elif self.cfg.vllm:
            output, (input_tokens_len, new_tokens) = self.__vllm_model(model, prompt, generation_config)
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
            elif self.cfg.vllm:
                output, (input_tokens_len, new_tokens) = self.__vllm_model(model, prompt, generation_config)
            else:
                output, (input_tokens_len, new_tokens) = self.__general_model(model, tokenizer, prompt, model_version, generation_config)
        return output
