from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from argparse import ArgumentParser
import random
import time
import re
from tqdm import tqdm
from sampler_hijack import hijack_samplers
import sys

total_token = 0
generation_time = 0
def add_token_cnt(cnt):
    global total_token
    total_token += cnt

def add_time(time):
    global generation_time
    generation_time += time

def get_novel_text_list(data_path, text_length):
    data_list = list()
    with open(data_path, 'r', encoding="utf-8") as f:
        data = f.read()
    data = data.strip()
    data_raw = re.sub('\n+', '\n', data)
    print(f"text total words: {len(data_raw)}")
    data = data_raw.strip().split("\n")
    i = 0
    while i < len(data):
        r = text_length
        text = ""
        while len(text) < r:
            if i >= len(data):
                break
            if len(text) > max(- len(data[i]) + r, 0):
                break
            else:
                text += data[i] + "\n"
                i += 1
        text = text.strip()
        data_list.append(text)
    return data_raw, data_list

def get_prompt(input, model_version):
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

def get_llm_sharp_type(model_version):
    if model_version == '0.5' or model_version == '0.8':
        return "BaichuanAwq"
    if model_version == '0.7':
        return "QwenAwq"
    if model_version == '0.1' or model_version == '0.4':
        return "Llama"

    raise ValueError(f"Wrong model version{model_version}, please view https://huggingface.co/sakuraumi/Sakura-13B-Galgame")

def split_response(response, model_version):
    response = response.replace("</s>", "")
    if model_version == '0.5' or model_version == '0.8':
        output = response.split("<reserved_107>")[1]
        return output
    if model_version == '0.7':
        output = response.split("<|im_start|>assistant\n")[1]
        return output
    if model_version == '0.1':
        output = response.split("\n\nAssistant: \n")[1]
        return output
    if model_version == '0.4':
        output = response.split("\nAssistant: ")[1]
        return output
    
    raise ValueError(f"Wrong model version{model_version}, please view https://huggingface.co/sakuraumi/Sakura-13B-Galgame")

def detect_degeneration(generation: list, model_version, text_length):
    if model_version != "0.8":
        return False
    i = generation.index(196)
    generation = generation[i+1:]
    print(len(generation))
    if len(generation) >= text_length:
        print("model degeneration detected, retrying...")
        return True
    else:
        return False

def get_model_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig, text_length: int, llama_cpp: bool, use_llm_sharp: bool):

    backup_generation_config_stage2 = GenerationConfig(
            temperature=0.1,
            top_p=0.3,
            top_k=40,
            num_beams=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=text_length,
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
            max_new_tokens=text_length,
            min_new_tokens=1,
            do_sample=True,
            repetition_penalty=1.0,
            frequency_penalty=0.2
        )


    backup_generation_config = [backup_generation_config_stage2, backup_generation_config_stage3]

    if llama_cpp:
        
        def generate(model, generation_config):
            if "frequency_penalty" in generation_config.__dict__.keys():
                output = model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'], temperature=generation_config.__dict__['temperature'], top_p=generation_config.__dict__['top_p'], repeat_penalty=generation_config.__dict__['repetition_penalty'], frequency_penalty=generation_config.__dict__['frequency_penalty'])
            else:
                output = model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'], temperature=generation_config.__dict__['temperature'], top_p=generation_config.__dict__['top_p'], repeat_penalty=generation_config.__dict__['repetition_penalty'])
            return output
        
        stage = 0
        output = generate(model, generation_config)
        while output['usage']['completion_tokens'] == text_length:
            stage += 1
            if stage > 2:
                print("model degeneration cannot be avoided.")
                break
            print("model degeneration detected, retrying...")
            output = generate(model, backup_generation_config[stage-1])
        response = output['choices'][0]['text']
        return response

    elif use_llm_sharp:
        import System
        import llm_sharp
        def generate(model, generation_config):
            history = System.Collections.Generic.List[System.ValueTuple[System.String, System.String]]()
            g = llm_sharp.LLM.Pretrained.GenerationConfig()
            g.temperature = generation_config.__dict__['temperature']
            g.top_p = generation_config.__dict__['top_p']
            g.max_generated_tokens = generation_config.__dict__['max_new_tokens']
            output = model.chat(history, prompt, g)
            output_ret = ""
            cnt = 0
            for o in output:
                output_ret += o
                cnt += 1
            add_token_cnt(cnt)
            return output_ret
            
        response = generate(model, generation_config)
        return response

    generation = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), generation_config=generation_config)[0]
    if len(generation) > text_length:
        stage = 0
        while detect_degeneration(list(generation), model_version, text_length):
            stage += 1
            if stage > 2:
                print("model degeneration cannot be avoided.")
                break
            generation = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), generation_config=backup_generation_config[stage-1])[0]
    response = tokenizer.decode(generation)
    output = split_response(response, model_version)
    return output

def get_compare_text(source_text, translated_text):
    source_text_list = source_text.strip().split("\n")
    translated_text_list = translated_text.strip().split("\n")
    output_text = ""
    if len(source_text_list) != len(translated_text_list):
        print(f"error occurred when output compared text(length of source is {len(source_text_list)} while length of translated is {len(translated_text_list)}), fallback to output only translated text.")
        # for i in range(len(source_text_list)):
        #     try:
        #         tmp = translated_text_list[i]
        #     except Exception as e:
        #         tmp = ""
        #     output_text += source_text_list[i] + "\n" + tmp + "\n\n"
        return translated_text
    else:
        for i in range(len(source_text_list)):
            output_text += source_text_list[i] + "\n" + translated_text_list[i] + "\n\n"
        output_text = output_text.strip()
        return output_text
    


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="SakuraLLM/Sakura-13B-LNovel-v0.8", help="model huggingface id or local path.")
    parser.add_argument("--use_gptq_model", action="store_true", help="whether your model is gptq quantized.")
    parser.add_argument("--model_version", type=str, default="0.8", help="model version written on huggingface readme, now we have ['0.1', '0.4', '0.5', '0.7', '0.8']")
    parser.add_argument("--data_path", type=str, default="data.txt", help="file path of the text you want to translate.")
    parser.add_argument("--output_path", type=str, default="data_translated.txt", help="save path of the text model translated.")
    parser.add_argument("--text_length", type=int, default=512, help="input max length in each inference.")
    parser.add_argument("--compare_text", action="store_true", help="whether to output with both source text and translated text in order to compare.")
    parser.add_argument("--trust_remote_code", action="store_true", help="whether to trust remote code.")
    parser.add_argument("--llama", action="store_true", help="whether your model is llama family.")
    parser.add_argument("--llama_cpp", action="store_true", help="whether to use llama.cpp.")
    parser.add_argument("--use_gpu", action="store_true", help="whether to use gpu when using llama.cpp.")
    parser.add_argument("--n_gpu_layers", type=int, default=0, help="layers cnt when using gpu in llama.cpp")
    parser.add_argument("--use_llm_sharp", action="store_true", help="whether to use llm-sharp.")
    parser.add_argument("--llm_sharp_lib_path", type=str, default="./llm-sharp", help="path of the library of llm-sharp")
    args = parser.parse_args()

    if args.use_gptq_model + args.llama_cpp + args.use_llm_sharp > 1:
        raise ValueError('You are using two or more of the "use_gptq_model", "llama_cpp", "llm_sharp" flags, which is not supported.')
    
    if args.use_gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
    
    if args.llama_cpp:            
        from llama_cpp import Llama
        
    if args.use_llm_sharp:
        path = args.llm_sharp_lib_path
        sys.path.append(path)
        from pythonnet import load
        load("coreclr")
        import clr
        clr.AddReference("LLM")
        from llm_sharp.LLM.Pretrained import LanguageModel
        from llm_sharp.LLM.Utils import LibTorchLoader
        from TorchSharp import torch
        
    if args.llama:
        from transformers import LlamaForCausalLM, LlamaTokenizer
        
    if args.trust_remote_code is False and args.model_version in "0.5 0.7 0.8":
        
        raise ValueError("If you use model version 0.5, 0.7 or 0.8, please add flag --trust_remote_code.")

    hijack_samplers()
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.3,
        top_k=40,
        num_beams=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=512,
        min_new_tokens=1,
        do_sample=True
    )    

    print("loading...")
    if not args.llama_cpp and not args.use_llm_sharp:
        if args.llama:
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=args.trust_remote_code)
    else:
        tokenizer = None

    if args.use_gptq_model:
        model = AutoGPTQForCausalLM.from_quantized(args.model_name_or_path, device="cuda:0", trust_remote_code=args.trust_remote_code)
    elif args.llama:
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=args.trust_remote_code)
    elif args.llama_cpp:
        if args.use_gpu:
            n_gpu = -1 if args.n_gpu_layers == 0 else args.n_gpu_layers
        else:
            n_gpu = 0
        model = Llama(model_path=args.model_name_or_path, n_gpu_layers=n_gpu, n_ctx=4 * args.text_length)
    elif args.use_llm_sharp:
        LibTorchLoader.EnsureLoaded()
        device = torch.device("cuda")
        dtype = torch.ScalarType.Float16
        model = LanguageModel.from_pretrained(get_llm_sharp_type(args.model_version), args.model_name_or_path, dtype, device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=args.trust_remote_code)
            
    print("translating...")
    start = time.time()

    data_raw, data_list = get_novel_text_list(args.data_path, args.text_length)
    data = ""
    for d in tqdm(data_list):
        prompt = get_prompt(d, args.model_version)
        output = get_model_response(model, tokenizer, prompt, args.model_version, generation_config, args.text_length, args.llama_cpp, args.use_llm_sharp)
        data += output.strip() + "\n"

    end = time.time()
    print("translation completed, used time: ", end-start, ", total tokens: ", total_token, ", speed: ", total_token/(end-start), " token/s")

    print("saving...")
    if args.compare_text:
        with open(args.output_path, 'w', encoding='utf-8') as f_w:
            f_w.write(get_compare_text(data_raw, data))
    else:
        with open(args.output_path, 'w', encoding='utf-8') as f_w:
            f_w.write(data)

    print("completed.")

if __name__ == "__main__":

    main()
