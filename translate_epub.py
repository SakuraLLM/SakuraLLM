from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from argparse import ArgumentParser
import time
import os, re
import fnmatch
import glob
import shutil
import zipfile
from tqdm import tqdm
from sampler_hijack import hijack_samplers

def find_all_htmls(root_dir):
    html_files = []
    for foldername, subfolders, filenames in os.walk(root_dir):
        for extension in ['*.html', '*.xhtml', '*.htm']:
            for filename in fnmatch.filter(filenames, extension):
                file_path = os.path.join(foldername, filename)
                html_files.append(file_path)
    return html_files

def get_html_text_list(epub_path, text_length):
    data_list = []

    def clean_text(text):
        text=re.sub(r'<rt[^>]*?>.*?</rt>', '', text)
        text=re.sub(r'<[^>]*>|\n', '', text)
        return text

    with open(epub_path, 'r', encoding='utf-8') as f:
        file_text = f.read()
        matches = re.finditer(r'<(h[1-6]|p).*?>(.+?)</\1>', file_text, flags=re.DOTALL)
        if not matches:
            print("perhaps this file is a struct file")
            return data_list, file_text
        groups = []
        text = ''
        pre_end = 0
        for match in matches:
            if len(text + match.group(2)) <= text_length:
                new_text = clean_text(match.group(2))
                if new_text:
                    groups.append(match)
                    text += '\n' + new_text
            else:
                data_list.append((text, groups, pre_end))
                pre_end = groups[-1].end()
                new_text = clean_text(match.group(2))
                if new_text:
                    groups = [match]
                    text = clean_text(match.group(2))
                else:
                    groups = []
                    text = ''

        if text:
            data_list.append((text, groups, pre_end))
    # TEST:
    # for d in data_list:
    #     print(f"{len(d[0])}", end=" ")
    return data_list, file_text

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
    if len(generation) >= 1023:
        print("model degeneration detected, retrying...")
        return True
    else:
        return False

def get_model_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig, text_length: int, llama_cpp: bool):

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

    if llama_cpp:
        output = model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'], temperature=generation_config.__dict__['temperature'], top_p=generation_config.__dict__['top_p'], repeat_penalty=generation_config.__dict__['repetition_penalty'])
        response = output['choices'][0]['text']
        return response

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


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="SakuraLLM/Sakura-13B-LNovel-v0.8", help="model huggingface id or local path.")
    parser.add_argument("--use_gptq_model", action="store_true", help="whether your model is gptq quantized.")
    parser.add_argument("--model_version", type=str, default="0.8", help="model version written on huggingface readme, now we have ['0.1', '0.4', '0.5', '0.7', '0.8']")
    parser.add_argument("--data_path", type=str, default="", help="file path of the epub you want to translate.")
    parser.add_argument("--data_folder", type=str, default="", help="folder path of the epubs you want to translate.")
    parser.add_argument("--output_folder", type=str, default="", help="save folder path of the epubs model translated.")
    parser.add_argument("--text_length", type=int, default=512, help="input max length in each inference.")
    parser.add_argument("--trust_remote_code", action="store_true", help="whether to trust remote code.")
    parser.add_argument("--llama", action="store_true", help="whether your model is llama family.")
    parser.add_argument("--llama_cpp", action="store_true", help="whether to use llama.cpp.")
    parser.add_argument("--use_gpu", action="store_true", help="whether to use gpu when using llama.cpp.")
    parser.add_argument("--n_gpu_layers", type=int, default=0, help="layers cnt when using gpu in llama.cpp")
    args = parser.parse_args()

    if args.use_gptq_model:
        from auto_gptq import AutoGPTQForCausalLM

    if args.llama_cpp:
        if args.use_gptq_model:
            raise ValueError("You are using both use_gptq_model and llama_cpp flag, which is not supported.")
        from llama_cpp import Llama

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

    print("Loading model...")
    if not args.llama_cpp:
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
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=args.trust_remote_code)

    print("Start translating...")
    start = time.time()

    epub_list = []
    save_list = []
    if args.data_path:
        epub_list.append(args.data_path)
        save_list.append(os.path.join(args.output_folder, os.path.basename(args.data_path)))
    if args.data_folder:
        os.makedirs(args.output_folder, exist_ok=True)
        for f in os.listdir(args.data_folder):
            if f.endswith(".epub"):
                epub_list.append(os.path.join(args.data_folder, f))
                save_list.append(os.path.join(args.output_folder, f))

    for epub_path, save_path in zip(epub_list, save_list):
        print(f"translating {epub_path}...")
        start_epub = time.time()

        if os.path.exists('./temp'):
            shutil.rmtree('./temp')
        with zipfile.ZipFile(epub_path, 'r') as f:
            f.extractall('./temp')

        for html_path in find_all_htmls('./temp'):
            print(f"\ttranslating {html_path}...")
            start_html = time.time()

            translated = ''
            data_list, file_text = get_html_text_list(html_path, args.text_length)
            if len(data_list) == 0:
                    continue
            for text, groups, pre_end in tqdm(data_list):
                prompt = get_prompt(text, args.model_version)
                output = get_model_response(model, tokenizer, prompt, args.model_version, generation_config, args.text_length, args.llama_cpp)
                texts = output.strip().split('\n')
                if len(texts) < len(groups):
                    texts += [''] * (len(groups) - len(texts))
                else:
                    texts = texts[:len(groups)-1] + ['<br/>'.join(texts[len(groups)-1:])]
                for t, match in zip(texts, groups):
                    t = match.group(0).replace(match.group(2), t)
                    translated += file_text[pre_end:match.start()] + t
                    pre_end = match.end()

            translated += file_text[data_list[-1][1][-1].end():]
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(translated)

            end_html = time.time()
            print(f"\t{html_path} translated, used time: ", end_html-start_html)

        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as f:
            for file_path in glob.glob(f'./temp/**', recursive=True):
                if not os.path.isdir(file_path):
                    relative_path = os.path.relpath(file_path, './temp')
                    f.write(file_path, relative_path)
        shutil.rmtree('./temp')

        end_epub = time.time()
        print(f"{epub_path} translated, used time: ", end_epub-start_epub)

    end = time.time()
    print("translation completed, used time: ", end-start)

if __name__ == "__main__":
    main()
