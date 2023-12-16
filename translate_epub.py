from dacite import from_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import os, re
import fnmatch
import glob
import shutil
import zipfile
from tqdm import tqdm

import utils
import utils.cli
import utils.model as M
import utils.consts as consts

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
        matches = re.finditer(r'<(h[1-6]|p|a|title).*?>(.+?)</\1>', file_text, flags=re.DOTALL)
        if not matches:
            print("perhaps this file is a struct file")
            return data_list, file_text
        groups = []
        text = ''
        pre_end = 0
        for match in matches:
            match_text = clean_text(match.group(2))
            # 第一次强制走if分支，确保一定有至少一条文本。
            if len(text + match_text) <= text_length or text == '':
                new_text = match_text
                if new_text:
                    groups.append(match)
                    text += '\n' + new_text
            else:
                data_list.append((text, groups, pre_end))
                pre_end = groups[-1].end()
                new_text = match_text
                if new_text:
                    groups = [match]
                    text = match_text
                else:
                    groups = []
                    text = ''

        if text:
            data_list.append((text, groups, pre_end))
    # TEST:
    # for d in data_list:
    #     print(f"{len(d[0])}", end=" ")
    return data_list, file_text


def get_model_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, model_version: str, generation_config: GenerationConfig, text_length: int, llama_cpp: bool):
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

    generation = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), generation_config=generation_config)[0]
    if len(generation) > text_length:
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


def main():
    def extra_args(parser):
        parser.add_argument("--data_path", type=str, default="", help="file path of the epub you want to translate.")
        parser.add_argument("--data_folder", type=str, default="", help="folder path of the epubs you want to translate.")
        parser.add_argument("--output_folder", type=str, default="", help="save folder path of the epubs model translated.")
        parser.add_argument("--text_length", type=int, default=512, help="input max length in each inference.")
        parser.add_argument("--translate_title", action='store_true', help='whether to translate the file names of the epubs')

    args = utils.cli.parse_args(do_validation=True, add_extra_args_fn=extra_args)

    import coloredlogs
    coloredlogs.install(level="INFO")

    cfg = from_dict(data_class=M.SakuraModelConfig, data=args.__dict__)
    sakura_model = M.SakuraModel(cfg=cfg)

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

    print("Start translating...")
    start = time.time()

    epub_list = []
    save_list = []
    if args.data_path:
        assert args.data_path.endswith(".epub")
        epub_list.append(args.data_path)
        f = os.path.basename(args.data_path)
        if args.translate_title:
            prompt = consts.get_prompt(
                input=f[:-5], 
                model_name=sakura_model.cfg.model_name,
                model_version=sakura_model.cfg.model_version,
                model_quant=sakura_model.cfg.model_quant
            )
            output = get_model_response(
                sakura_model.model,
                sakura_model.tokenizer,
                prompt,
                sakura_model.cfg.model_version,
                generation_config,
                sakura_model.cfg.text_length,
                sakura_model.cfg.llama_cpp,
            )
            f = output.strip() + '.epub'
        save_list.append(os.path.join(args.output_folder, f))
    if args.data_folder:
        os.makedirs(args.output_folder, exist_ok=True)
        for f in os.listdir(args.data_folder):
            if f.endswith(".epub"):
                epub_list.append(os.path.join(args.data_folder, f))
                if args.translate_title:
                    prompt = consts.get_prompt(
                        input=f[:-5], 
                        model_name=sakura_model.cfg.model_name,
                        model_version=sakura_model.cfg.model_version,
                        model_quant=sakura_model.cfg.model_quant
                    )
                    output = get_model_response(
                        sakura_model.model,
                        sakura_model.tokenizer,
                        prompt,
                        sakura_model.cfg.model_version,
                        generation_config,
                        sakura_model.cfg.text_length,
                        sakura_model.cfg.llama_cpp,
                    )
                    f = output.strip() + '.epub'
                save_list.append(os.path.join(args.output_folder, f))

    for epub_path, save_path in zip(epub_list, save_list):
        print(f"translating {epub_path}...")
        start_epub = time.time()

        if os.path.exists('./temp'):
            shutil.rmtree('./temp')
        with zipfile.ZipFile(epub_path, 'r') as f:
            f.extractall('./temp')

        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as f:
            for html_path in find_all_htmls('./temp'):
                print(f"\ttranslating {html_path}...")
                start_html = time.time()

                translated = ''
                data_list, file_text = get_html_text_list(html_path, args.text_length)
                if len(data_list) == 0:
                        continue
                for text, groups, pre_end in tqdm(data_list):
                    prompt = consts.get_prompt(
                        input=text,
                        model_name=sakura_model.cfg.model_name,
                        model_version=sakura_model.cfg.model_version,
                        model_quant=sakura_model.cfg.model_quant,
                    )
                    #FIXME(kuriko): refactor this to sakura_model.completion()
                    output = get_model_response(
                        sakura_model.model,
                        sakura_model.tokenizer,
                        prompt,
                        sakura_model.cfg.model_version,
                        generation_config,
                        sakura_model.cfg.text_length,
                        sakura_model.cfg.llama_cpp,
                    )
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
                with open(html_path, 'w', encoding='utf-8') as fout:
                    fout.write(translated)

                end_html = time.time()
                print(f"\t{html_path} translated, used time: ", end_html-start_html)

            for file_path in glob.glob(f'./temp/**', recursive=True):
                if not os.path.isdir(file_path):
                    relative_path = os.path.relpath(file_path, './temp')
                    f.write(file_path, relative_path)

        shutil.rmtree('./temp')

        end_epub = time.time()
        print(f"{epub_path} translated, used time: ", end_epub-start_epub)

    end = time.time()
    print("translation completed, used time: ", end-start)


def test():
    path = "./temp/item/xhtml/p-009.xhtml"
    data_list, file_text = get_html_text_list(path, 512)

if __name__ == "__main__":
    main()
