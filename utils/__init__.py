import logging
from pprint import pprint, pformat
from transformers import GenerationConfig

logger = logging.getLogger(__name__)


def split_response(response, model_version):
    response = response.replace("</s>", "")
    if model_version == '0.5' or '0.8' in model_version:
        output = response.split("<reserved_107>")[1]
        return output
    if '0.7' in model_version or '0.9' in model_version:
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
    #TODO: refactor this
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


def log_generation_config(generation_config: GenerationConfig):
    logger.debug(f"current generation config: \n{pformat(generation_config.to_diff_dict())}")