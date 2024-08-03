from argparse import ArgumentParser
from dacite import from_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import re
from tqdm import tqdm
from pathlib import Path

import utils
import utils.cli
import utils.model as M
import utils.consts as consts

import copy
import pysubs2
import opencc
st_converter = opencc.OpenCC('s2t.json')

def get_subtitle_text_list(data_path):
    data_list = list()
    subs = pysubs2.load(data_path)
    subs.sort()
    # combaine subtitles with same timestamps
    for i in range(1, len(subs)):
        if (subs[i].start == subs[i-1].start):
            if (subs[i].end == subs[i-1].end):
                subs[i].text = " ".join((subs[i-1].text, subs[i].text))
                subs[i-1].text = ""
    subs.remove_miscellaneous_events()

    # combaine same subtitle line to timestamps
    for j in range(1, len(subs)):
        if (subs[j].text == subs[j-1].text):
            if (subs[j].start == subs[j-1].end):
                subs[j].start = subs[j-1].start
                subs[j-1].text = ""
    subs.remove_miscellaneous_events()

    for k in range(len(subs)):
        text = subs[k].text
        text = re.sub(r"\\N", " ", text)
        subs[k].text = text
        data_list.append(text)
    return subs, data_list

def set_styles(subs):
    default_style = pysubs2.SSAStyle()
    default_style.fontname = "NotoSansCJK"
    default_style.fontsize = 20
    default_style.outline = "1"
    default_style.shadow = "0.6"
    default_style.backcolor = "&H00868686"
    subs.styles["Default"] = default_style
    return subs

def save_subtitle(subs, data_path, data, lang_code):
    data_path = Path(data_path)
    data = data.strip()
    data = data.split("\n")
    subs = set_styles(subs)

    for i in range(len(subs)):
        subs[i].text = data[i] + "\\N{\\fs12}" + subs[i].text

    subs.save(data_path.stem + lang_code + ".ass")

    for j in range(len(subs)):
        new = subs[j].text
        subs[j].text = new.split("\\N")[0]

    subs.save(data_path.stem + lang_code + ".srt")
    return


total_token = 0
generation_time = 0
def add_token_cnt(cnt):
    global total_token
    total_token += cnt

def add_time(time):
    global generation_time
    generation_time += time

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


def main():
    def extra_args(parser: ArgumentParser):
        novel_group = parser.add_argument_group("Subtitles")
        novel_group.add_argument("--data_path", type=str, default="", help="file path of the subtitle you want to translate.")
        novel_group.add_argument("--save_traditional", default=False, action="store_true", help="whether to save Traditional Chinese subtitle.")
        novel_group.add_argument("--text_length", type=int, default=512, help="input max length in each inference.")

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

    print("translating...")
    start = time.time()

    subs, data_list = get_subtitle_text_list(args.data_path)
    data = ""
    for d in tqdm(data_list):
        prompt = consts.get_prompt(
            input=d,
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
    print("Translation completed, used time: ", generation_time, end-start, ", total tokens: ", total_token, ", speed: ", total_token/(end-start), " token/s")

    print("Saving...")

    tsubs = copy.deepcopy(subs)
    save_subtitle(subs, args.data_path, data, "chs")
    if (args.save_traditional):
        data = st_converter.convert(data)
        save_subtitle(tsubs, args.data_path, data, "cht")

    print("Completed.")

if __name__ == "__main__":
    main()
