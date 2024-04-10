import asyncio
import logging

from transformers import GenerationConfig, PretrainedConfig
from . import BaseInferEngine
from vllm import AsyncEngineArgs, AsyncLLMEngine, LLM, SamplingParams
from vllm.utils import Counter

from utils.model import SakuraModelConfig

logger = logging.getLogger(__name__)


class MixLLMEngine(LLM):
    "an AsyncLLMEngine unwrapper for flexible generation"

    def __init__(self, args: SakuraModelConfig):
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
        llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.llm_engine = llm_engine.engine
        self.async_engine = llm_engine
        self.request_counter = Counter()
        self.loop = asyncio.new_event_loop()
        self.req_id = 0

    def get_metadata(self, _: SakuraModelConfig):
        config = PretrainedConfig.from_pretrained(self.cfg.model_name_or_path)
        model_name = self.model.config.sakura_name
        model_quant = self.model.config.sakura_quant
        model_version = self.model.config.sakura_version
        return model_name, model_quant, model_version

    def __stream_generate(self, prompt, sampling_params):
        # NOTE(kuriko): when multi-requests come, the same `request-id` will cause a 500 internal error
        self.req_id += 1
        generator = self.async_engine.generate(prompt, sampling_params, request_id=f"Sakura-{self.req_id}")
        while True:
            try:
                output = self.loop.run_until_complete(anext(generator))
                yield output
            except StopAsyncIteration:
                break

    def __generate(self, prompt, sampling_params, stream=False):
        if stream:
            return self.__stream_generate(prompt, sampling_params)
        else:
            return super(MixLLMEngine, self).generate(prompt, sampling_params)

    def generate(self, prompt: str, generation_config: GenerationConfig):
        logger.debug(f"prompt is: {prompt}")
        sampling_params = SamplingParams(
            max_tokens=generation_config.__dict__['max_new_tokens'],
            temperature=generation_config.__dict__['temperature'],
            top_p=generation_config.__dict__['top_p'],
            repetition_penalty=generation_config.__dict__['repetition_penalty'],
            frequency_penalty=generation_config.__dict__['frequency_penalty'],
        )
        # need to unwrap the async engine for generation
        output = self.__generate(prompt, sampling_params)
        request_output = output[0].outputs[0]
        text = request_output.text
        input_tokens_len = len(output[0].prompt_token_ids)
        new_tokens = len(request_output.token_ids)
        return text, (input_tokens_len, new_tokens)

    def stream_generate(self, prompt: str, generation_config: GenerationConfig):
        logger.debug(f"prompt is: {prompt}")
        sampling_params = SamplingParams(
            max_tokens=generation_config.__dict__['max_new_tokens'],
            temperature=generation_config.__dict__['temperature'],
            top_p=generation_config.__dict__['top_p'],
            repetition_penalty=generation_config.__dict__['repetition_penalty'],
            frequency_penalty=generation_config.__dict__['frequency_penalty'],
        )
        previous_output = ""
        for output in self.__generate(prompt, sampling_params, stream=True):
            output_text = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            delta_text = output_text.removeprefix(previous_output)
            previous_output = output_text
            yield delta_text, finish_reason
