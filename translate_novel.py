from argparse import ArgumentParser
from dacite import from_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import re
import os
from tqdm import tqdm

import utils
import utils.cli
import utils.model as M
import utils.consts as consts


total_token = 0
generation_time = 0
generation_config = None
sakura_model = None
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
                output = model.model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'], temperature=generation_config.__dict__['temperature'], top_p=generation_config.__dict__['top_p'], repeat_penalty=generation_config.__dict__['repetition_penalty'], frequency_penalty=generation_config.__dict__['frequency_penalty'])
            else:
                output = model.model(prompt, max_tokens=generation_config.__dict__['max_new_tokens'], temperature=generation_config.__dict__['temperature'], top_p=generation_config.__dict__['top_p'], repeat_penalty=generation_config.__dict__['repetition_penalty'])
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

    # llm sharp backend
    # elif use_llm_sharp:
    #     raise NotImplementedError
        # import System
        # import llm_sharp
        # def generate(model, generation_config):
        #     history = System.Collections.Generic.List[System.ValueTuple[System.String, System.String]]()
        #     g = llm_sharp.LLM.Pretrained.GenerationConfig()
        #     g.temperature = generation_config.__dict__['temperature']
        #     g.top_p = generation_config.__dict__['top_p']
        #     g.max_generated_tokens = generation_config.__dict__['max_new_tokens']
        #     output = model.chat(history, prompt, g)
        #     output_ret = ""
        #     cnt = 0
        #     for o in output:
        #         output_ret += o
        #         cnt += 1
        #     add_token_cnt(cnt)
        #     return output_ret

        # response = generate(model, generation_config)
        # return response

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

    # FIXME(kuriko): I dont know how refactor to this, QAQ. just provide an example.
    def get_model_response(model: M.SakuraModel, prompt: str, generation_config: GenerationConfig):
        backup_generation_config_stage2 = GenerationConfig( temperature=0.1, top_p=0.3, top_k=40, num_beams=1, bos_token_id=1, eos_token_id=2, pad_token_id=0, max_new_tokens=2 * text_length, min_new_tokens=1, do_sample=True, repetition_penalty=1.0, frequency_penalty=0.05)
        backup_generation_config_stage3 = GenerationConfig( temperature=0.1, top_p=0.3, top_k=40, num_beams=1, bos_token_id=1, eos_token_id=2, pad_token_id=0, max_new_tokens=2 * text_length, min_new_tokens=1, do_sample=True, repetition_penalty=1.0, frequency_penalty=0.2)

        backup_generation_config = [backup_generation_config_stage2, backup_generation_config_stage3]

        # Use the sync one
        output: M.SakuraModel.ModelResponse = model.completion(prompt, generation_config)
        if llama_cpp:
            return output.text

        # FIXME(kuriko): QAQ
        if len(generation) > 2 * text_length:
            stage = 0
            while utils.detect_degeneration(list(generation), model_version):
                stage += 1
                if stage > 2:
                    print("model degeneration cannot be avoided.")
                    break
                output: M.SakuraModel.ModelResponse = model.completion(prompt, backup_generation_config[stage-1])
        return output.text

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

def handle(output_path, data_path, text_length, compare_text):
    global sakura_model
    global generation_config
    with open(output_path, 'w', encoding='utf-8') as f_w:
        start = time.time()

        data_raw, data_list = get_novel_text_list(data_path, text_length)
        data = ""
        for d in tqdm(data_list):
            prompt = consts.get_prompt(
                raw_jp_text=d,
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
            data += output.strip() + "\n"

        end = time.time()
        print("translation completed, used time: ", generation_time, end-start, ", total tokens: ", total_token, ", speed: ", total_token/(end-start), " token/s")

        print("saving...")
        if compare_text:
            f_w.write(get_compare_text(data_raw, data))
        else:
            f_w.write(data)

def main():
    global sakura_model
    global generation_config
    def extra_args(parser: ArgumentParser):
        novel_group = parser.add_argument_group("Novel")
        novel_group.add_argument("--data_path", type=str, default="data.txt", help="file path of the text you want to translate.")
        novel_group.add_argument("--output_path", type=str, default="data_translated.txt", help="save path of the text model translated.")

        novel_group.add_argument("--data_folder", type=str, required=False, help="folder path of the text you want to translate.")
        novel_group.add_argument("--output_folder", type=str, required=False, help="save folder path of the text model translated.")

        novel_group.add_argument("--shutdown", action="store_true", help="shutdown after translate is completed")

        novel_group.add_argument("--compare_text", action="store_true", help="whether to output with both source text and translated text in order to compare.")
        novel_group.add_argument("--text_length", type=int, default=512, help="input max length in each inference.")

    args = utils.cli.parse_args(do_validation=True, add_extra_args_fn=extra_args)

    is_batch_process = False
    if args.data_folder and args.output_folder:
        is_batch_process = True
        print(f"Using batch mode:\n"
              f"Input Folder:  {args.data_folder}\n"
              f"Output Folder: {args.output_folder}")
    else:
        print(f"Using normal mode:\n"
              f"Input File:  {args.data_path}\n"
              f"Output File: {args.output_path}")

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

    print("translating...")
    if is_batch_process:
        os.makedirs(args.output_folder, exist_ok=True)
        for f in os.listdir(args.data_folder):
            if f.endswith(".txt") or f.endswith(".srt"):
                handle(os.path.join(args.output_folder, f), os.path.join(args.data_folder, f), args.text_length, args.compare_text)
    else:
        handle(args.output_path, args.data_path, args.text_length, args.compare_text)
    if args.shutdown == 1:
        os.system("/usr/bin/shutdown")
    print("completed.")

if __name__ == "__main__":

    main()
