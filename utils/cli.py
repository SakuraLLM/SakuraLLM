from argparse import ArgumentParser

def parse_args(do_validation:bool=False, add_extra_args_fn:any=None):
    # parse config
    parser = ArgumentParser()

    if add_extra_args_fn:
        add_extra_args_fn(parser)

    # log
    parser.add_argument("-l", "--log", dest="logLevel", choices=[
                        'trace', 'debug', 'info', 'warning', 'error', 'critical'], default="info", help="Set the logging level")

    # Model Path
    parser.add_argument("--model_name_or_path", type=str,
                        default="SakuraLLM/Sakura-13B-LNovel-v0.8", help="model huggingface id or local path.")

    # Model Version
    parser.add_argument("--model_version", type=str, default="0.8",
                        help="model version written on huggingface readme, now we have ['0.1', '0.4', '0.5', '0.7', '0.8', '0.9', '0.10']")

    # Model Quant Method
    quant_method_group = parser.add_argument_group("Quant Methods")
    quant_method_group.add_argument("--use_gptq_model", action="store_true", help="whether your model is gptq quantized.")
    quant_method_group.add_argument("--use_awq_model", action="store_true", help="whether your model is awq quantized.")

    # Others
    parser.add_argument("--trust_remote_code", action="store_true", help="whether to trust remote code.")

    # llama.cpp backend
    llamacpp_group = parser.add_argument_group("llama.cpp backend")
    llamacpp_group.add_argument("--llama_cpp", action="store_true", help="whether to use llama.cpp.")
    llamacpp_group.add_argument("--use_gpu", action="store_true", help="whether to use gpu when using llama.cpp.")
    llamacpp_group.add_argument("--n_gpu_layers", type=int, default=0, help="layers cnt when using gpu in llama.cpp")

    # vllm backend
    vllm_group = parser.add_argument_group("vllm backend")
    vllm_group.add_argument("--vllm", action="store_true", help="whether to use vllm.")
    vllm_group.add_argument("--enforce_eager", action="store_true", help="enable eager mode in vllm.")
    vllm_group.add_argument("--tensor_parallel_size", type=int, default=1, help="tensor parallel size when using gpu in vllm.")
    vllm_group.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="The ratio (between 0.0 and 1.0) of GPU memory for inference.")

    # ollama backend
    ollama_group = parser.add_argument_group("ollama backend")
    ollama_group.add_argument("--ollama", action="store_true", help="whether to use ollama.")

    # Deprecated
    abandon_group = parser.add_argument_group("Deprecated Options")
    abandon_group.add_argument("--llama", action="store_true", help="whether your model is llama family.")

    args = parser.parse_args()

    if do_validation:
        args_validation(args)

    return args


def args_validation(args) -> bool:
    # Simply verifiy the args herer, also add `imports` here to do some import before loading the packages
    if args.use_gptq_model:
        from auto_gptq import AutoGPTQForCausalLM

    if args.llama_cpp:
        if args.use_gptq_model:
            raise ValueError("You are using both use_gptq_model and llama_cpp flag, which is not supported.")
        from llama_cpp import Llama

    if args.llama:
        from transformers import LlamaForCausalLM, LlamaTokenizer

    if args.vllm:
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    if args.trust_remote_code is False and args.model_version in "0.5 0.7 0.8 0.9 0.10":
        args.trust_remote_code = True
        # raise ValueError("If you use model version 0.5, 0.7, 0.8 or 0.9, please add flag --trust_remote_code.")

    if args.use_gptq_model and args.use_awq_model:
        raise ValueError("You are using both use_gptq_model and use_awq_model flag, please specify only one quantization.")

    return True
