from transformers import GenerationConfig

def get_default_generation_config():
    return GenerationConfig(
        temperature=0.1,
        top_p=0.3,
        top_k=40,
        num_beams=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=1024,
        min_new_tokens=1,
        do_sample=True,
        # repetition_penalty=1.0,
        # frequency_penalty=0.2,
        # presence_penalty=0,
        # repetition_penalty_range=0,
    )

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