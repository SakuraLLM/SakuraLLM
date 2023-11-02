from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from argparse import ArgumentParser
import random
import time
import re
from tqdm import tqdm
from sampler_hijack import hijack_samplers

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

def detect_degeneration(generation: list, model_version):
    if model_version != "0.8":
        return False
    i = generation.index(196)
    generation = generation[i+1:]
    print(len(generation))
    if len(generation) >= 1023:
        print("model degeneration detected, retrying...")
        return True
    else:
        return False

def get_model_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig, text_length: int):

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
        while detect_degeneration(list(generation), model_version):
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
    args = parser.parse_args()

    if args.use_gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        
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
        max_new_tokens=1024,
        min_new_tokens=1,
        do_sample=True
    )    

    print("loading...")
    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=args.trust_remote_code)

    if args.use_gptq_model:
        model = AutoGPTQForCausalLM.from_quantized(args.model_name_or_path, device="cuda:0", trust_remote_code=args.trust_remote_code)
    else:
        if args.llama:
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=args.trust_remote_code)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=args.trust_remote_code)
            
    print("translating...")
    start = time.time()

    data_raw, data_list = get_novel_text_list(args.data_path, args.text_length)
    data = ""
    for d in tqdm(data_list):
        prompt = get_prompt(d, args.model_version)
        output = get_model_response(model, tokenizer, prompt, args.model_version, generation_config, args.text_length)
        data += output.strip() + "\n"

    end = time.time()
    print("translation completed, used time: ", generation_time, end-start, ", total tokens: ", total_token, ", speed: ", total_token/(end-start), " token/s")

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
