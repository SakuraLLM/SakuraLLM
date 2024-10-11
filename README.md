<div align="center">
<h1>
  SakuraLLM
</h1>
<center>
  <b>Sakura</b>: <b><ins>S</ins></b>FT <ins><b>A</b></ins>nd RLHF models using <ins><b>K</b></ins>nowledge of <ins><b>U</b></ins>niversal Character and <ins><b>R</b></ins>elationship <ins><b>A</b></ins>ttributes for Japanese to Chinese Translation in Light Novel & Galgame Domain.
</center>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/sakuraumi/Sakura-13B-Galgame" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/models/sakuraumi/Sakura-13B-Galgame" target="_blank">ModelScope</a>
</p>

# 目前Sakura发布的所有模型均采用[CC BY-NC-SA 4.0协议](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)，Sakura所有模型与其衍生模型均禁止任何形式的商用！Sakura系列所有模型皆仅供学习交流使用，开发者对使用Sakura模型造成的问题不负任何责任。

# 介绍

- 基于一系列开源大模型构建，在通用日文语料与轻小说/Galgame等领域的中日语料上进行继续预训练与微调，旨在提供开源可控可离线自部署的、ACGN风格的日中翻译模型。

- 新建了[TG交流群](https://t.me/+QMDKZyO9GV1kNDA1)，欢迎交流讨论。

**对于其他适配本模型的项目如使用非本项目提供的prompt格式进行翻译，不保证会获得与README中的说明一致的质量！**

**如果使用模型翻译并发布，请在最显眼的位置标注机翻！！！！！开发者对于滥用本模型造成的一切后果不负任何责任。**
> 由于模型一直在更新，请同时注明使用的模型版本等信息，方便进行质量评估和更新翻译。

**对于模型翻译的人称代词问题（错用，乱加，主宾混淆，男女不分等）和上下文理解问题，如果有好的想法或建议，欢迎提issue！**

### TODO：见https://github.com/SakuraLLM/Sakura-13B-Galgame/issues/42

## 快速开始

### 教程：

详见[本仓库Wiki](https://github.com/SakuraLLM/Sakura-13B-Galgame/wiki).

部分使用方法：[usage.md](https://github.com/SakuraLLM/SakuraLLM/blob/main/usage.md)

### 模型下载：

| 参数量 | 发布时间-底模-版本  | 模型 |
|:-------:|:-------|:-------|
| 32B | 20240508-Qwen1.5-32B-v0.9 | 🤗 [Sakura-32B-Qwen2beta-v0.9-GGUF](https://huggingface.co/SakuraLLM/Sakura-32B-Qwen2beta-v0.9-GGUF) |
|     | 20240508-Qwen1.5-32B-v0.10pre1 | 🤗 [Sakura-32B-Qwen2beta-v0.10pre1-GGUF](https://huggingface.co/SakuraLLM/Sakura-32B-Qwen2beta-v0.10pre1-GGUF) |
| 14B | 20240111-Qwen-14B-v0.9 | 🤗 [Sakura-13B-LNovel-v0.9b-GGUF](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.9b-GGUF) |
|     | 20240213-Qwen1.5-14B-v0.9 | 🤗 [Sakura-14B-Qwen2beta-v0.9-GGUF](https://huggingface.co/SakuraLLM/Sakura-14B-Qwen2beta-v0.9-GGUF) |
|     | 20240516-Qwen1.5-14B-v0.9.2 | 🤗 [Sakura-14B-Qwen2beta-v0.9.2-GGUF](https://huggingface.co/SakuraLLM/Sakura-14B-Qwen2beta-v0.9.2-GGUF)
|(最新)| **20241008-Qwen2.5-14B-v1.0** | 🤗 [Sakura-14B-Qwen2.5-v1.0-GGUF](https://huggingface.co/SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF)
| 7B  | 20240116-Qwen-7B-v0.9 | 🤗 [Sakura-7B-LNovel-v0.9-GGUF](https://huggingface.co/SakuraLLM/Sakura-7B-LNovel-v0.9-GGUF) |
|     | 20240531-Qwen1.5-7B-Galtransl-v2.6 | 🤗 [Galtransl-v2.6](https://huggingface.co/SakuraLLM/GalTransl-7B-v2.6) |
| ~2B | 20240214-Qwen1.5-1.8B-v0.9.1 | 🤗 [Sakura-1B8-Qwen2beta-v0.9.1-GGUF](https://huggingface.co/SakuraLLM/Sakura-1B8-Qwen2beta-v0.9.1-GGUF) |
|     | **20241012-Qwen2.5-1.5B-v1.0** | 🤗 [Sakura-1.5B-Qwen2.5-v1.0-GGUF](https://huggingface.co/SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF) |


p.s. 如果无法连接到HuggingFace服务器，可将链接中的`huggingface.co`改成`hf-mirror.com`，使用hf镜像站下载。

## News

1. **更新了基于Qwen2.5-14B的v1.0正式版模型[Sakura-14B-Qwen2.5-v1.0](https://huggingface.co/SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF)，prompt格式参见[下方说明](https://github.com/SakuraLLM/SakuraLLM#%E6%8E%A8%E7%90%86)。主要改进：**
    - 改善翻译质量，提高翻译准确率，尤其是人称的准确率。
    - 支持术语表(GPT字典)，以保持专有名词和人称的一致性。
    - 提高部分简单控制符的保留能力，尤其是单行内存在`\n`的情况下保留`\n`的能力。降低行数与原文不一致的概率。
    - 由于底模使用GQA，推理速度和显存占用显著改善，可实现更快的多线程推理。关于多线程推理，可参考[Sakura启动器GUI使用教程](https://books.fishhawk.top/forum/656d60530286f15e3384fcf8)或[SakuraLLMServer](https://github.com/neavo/SakuraLLMServer)。

1. **更新了基于Qwen1.5-7B的[Galtransl](https://huggingface.co/SakuraLLM/GalTransl-v1)模型，为视觉小说翻译任务专项优化。对视觉小说脚本中的行内换行、控制符、ruby注音等符号具有较好的保留能力。适配[GalTransl视觉小说翻译工具](https://github.com/xd2333/GalTransl)并调优，支持GPT字典（[字典写法见此](https://github.com/xd2333/GalTransl/wiki/GPT%E5%AD%97%E5%85%B8%E2%80%90sakura-galtransl)）。**

1. **增加了vllm模型后端的支持，详见**[#40](https://github.com/SakuraLLM/Sakura-13B-Galgame/pull/40)

1.  <del>感谢[Isotr0py](https://github.com/Isotr0py)提供运行模型的NoteBook仓库[SakuraLLM-Notebooks](https://github.com/Isotr0py/SakuraLLM-Notebooks)，可在[Colab](https://colab.research.google.com/)(免费T4\*1)与[Kaggle](https://www.kaggle.com/)(免费P100\*1或T4\*2)平台使用。**已经更新Kaggle平台的[使用教程](https://github.com/SakuraLLM/Sakura-13B-Galgame/wiki/%E7%99%BD%E5%AB%96Kaggle%E5%B9%B3%E5%8F%B0%E9%83%A8%E7%BD%B2%E6%95%99%E7%A8%8B)，可以白嫖一定时间的T4\*2。**</del>
   警告，Kaggle 官方已经采取措施封禁 SakuraLLM 所有模型，[参见](https://github.com/SakuraLLM/SakuraLLM/issues/115) ，在 Kaggle 上使用 SakuraLLM 将会导致永久性封号。请转移至租卡或者利用机翻站算力共享工具（为防止滥用，请自行搜索）。
  
1.  **Sakura API已经支持OpenAI格式，现在可以通过OpenAI库或者OpenAI API Reference上的请求形式与Server交互。**
一个使用OpenAI库与Sakura模型交互的例子详见[openai_example.py](https://github.com/SakuraLLM/Sakura-13B-Galgame/blob/main/tests/example_openai.py)。

## 已经接入模型的工具

1. 网站：[轻小说机翻机器人](https://books.fishhawk.top/)已接入Sakura模型(v0.8-4bit)，站内有大量模型翻译结果可供参考。你也可以自行部署模型并使用该网站生成机翻，目前已经支持v0.8与v0.9模型，且提供了llama.cpp一键包。
  
   轻小说机翻机器人网站是一个自动生成轻小说机翻并分享的网站。你可以浏览日文网络小说，或者上传Epub/Txt文件，并生成机翻。

1. [LunaTranslator](https://github.com/HIllya51/LunaTranslator)已经支持Sakura API，可以通过本地部署API后端，并在LunaTranslator中配置Sakura API来使用Sakura模型进行Galgame实时翻译。  
    ~~使用[KurikoMoe](https://github.com/kurikomoe/LunaTranslator/releases/latest)的版本可以支持流式输出。~~ 目前官方版本已经支持流式输出，只需在翻译设置界面勾选流式输出即可。

   LunaTranslator是一个Galgame翻译工具，支持剪贴板、OCR、HOOK，支持40余种翻译引擎。

1. [GalTransl](https://github.com/XD2333/GalTransl)已经支持Sakura API，可以通过本地部署API后端，在GalTransl中配置使用Sakura模型来翻译Galgame，制作内嵌式翻译补丁。

   GalTransl是一个galgame自动化翻译工具，用于制作内嵌式翻译补丁。一个使用GalTransl和Sakura模型翻译的[示例](https://www.ai2moe.org/files/file/2271-%E6%88%AF%E7%94%BBgaltranslsakuragpt35%E7%88%B1%E4%B9%8B%E5%90%BB3-sexy-gpt%E7%BF%BB%E8%AF%91%E8%A1%A5%E4%B8%81uploadee5-mb/)

1. 翻译Unity引擎游戏的工具[SakuraTranslator](https://github.com/fkiliver/SakuraTranslator)。感谢[fkiliver](https://github.com/fkiliver)提供。

1. 翻译RPGMaker引擎游戏的工具[RPGMaker_LLaMA_Translator](https://github.com/fkiliver/RPGMaker_LLaMA_Translator)。感谢[fkiliver](https://github.com/fkiliver)提供。

1. [AiNiee](https://github.com/NEKOparapa/AiNiee-chatgpt)已经支持Sakura API，可以通过本地部署API后端，在AiNiee中使用Sakura模型进行翻译。

   AiNiee是一款基于【mtool】或【Translator++】，chatgpt自动批量翻译工具，主要是用来翻译各种RPG游戏。

1. [manga-image-translator](https://github.com/zyddnys/manga-image-translator)已经支持Sakura API，可以通过本地部署API后端，使用Sakura自动翻译漫画。

1. [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)已经支持Sakura API，可以通过本地部署API后端，使用Sakura翻译漫画。

# 显存需求

下面的表格显示了使用不同量化和不同格式的模型时显存占用的大小。如果你的显卡显存不满足上述需求，可以尝试同时使用CPU与GPU进行推理。

- llama.cpp GGUF模型（使用Qwen-14B v0.9模型进行测试）

|  模型量化类型  | 模型大小 | 推荐显存大小 |
|:-------:|:-------:|:-------:|
| fp16 | 26.3G | 超出游戏显卡显存范围 |
| Q8_0 | 14G | 24G |
| Q6_K | 11.4G | 20G |
| Q5_K_M | 10.1G | 16G |
| Q4_K_M | 8.8G | 16G |
| Q3_K_M | 7.2G | 16G |
| Q2_K | 6.1G | 12G |

# 模型详情

## 描述

- Finetuned by [SakuraUmi](https://github.com/pipixia244)
- Finetuned on [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
- Continual Pre-trained on [Qwen model series](https://github.com/QwenLM/Qwen)
- Continual Pre-trained on [Qwen1.5 model series](https://github.com/QwenLM/Qwen1.5)
- Finetuned on Sakura-Base model series
- Languages: Chinese/Japanese

## 效果

- Galgame

  [一个例子](https://www.ai2moe.org/files/file/2271-%E6%88%AF%E7%94%BBgaltranslsakuragpt35%E7%88%B1%E4%B9%8B%E5%90%BB3-sexy-gpt%E7%BF%BB%E8%AF%91%E8%A1%A5%E4%B8%81uploadee5-mb/)
  
- 轻小说

  网站：[轻小说机翻机器人](https://books.fishhawk.top/)已接入Sakura模型(v0.9)，站内有大量模型翻译的轻小说可供参考。

- PPL

  Sakura-14B-Qwen2beta-v0.9-iq4_xs_ver2: 4.43
  
  Sakura-32B-Qwen2beta-v0.9-iq4xs: 3.28

# 推理

- openai api messages格式：

  - v0.9
    使用代码处理如下：
    ```python
    input_text_list = ['a', 'bb', 'ccc', ...] # 一系列上下文文本，每个元素代表一行的文本
    raw_text = "\n".join(input_text_list)
    messages=[
        {
            "role": "system",
            "content": "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
        },
        {
            "role": "user",
            "content": "将下面的日文文本翻译成中文：" + raw_text
        }
    ]
    ```
- prompt格式：

  - v1.0pre1
    代码处理如下：
    ```python
            gpt_dict = [{
              "src": "原文1",
              "dst": "译文1",
              "info": "注释信息1",
            },]
            gpt_dict_text_list = []
            for gpt in gpt_dict:
                src = gpt['src']
                dst = gpt['dst']
                info = gpt['info'] if "info" in gpt.keys() else None
                if info:
                    single = f"{src}->{dst} #{info}"
                else:
                    single = f"{src}->{dst}"
                gpt_dict_text_list.append(single)

            gpt_dict_raw_text = "\n".join(gpt_dict_text_list)

            user_prompt = "根据以下术语表（可以为空）：\n" + gpt_dict_raw_text + "\n" + "将下面的日文文本根据对应关系和备注翻译成中文：" + japanese
            prompt = "<|im_start|>system\n你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。<|im_end|>\n" \ # system prompt
            + "<|im_start|>user\n" + user_prompt + "<|im_end|>\n" \ # user prompt
            + "<|im_start|>assistant\n" # assistant prompt start

            # 如果术语表为空，也可以使用如下prompt（在术语表为空时更加推荐）
            user_prompt = "将下面的日文文本翻译成中文：" + japanese
            prompt = "<|im_start|>system\n你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。<|im_end|>\n" \ # system prompt
            + "<|im_start|>user\n" + user_prompt + "<|im_end|>\n" \ # user prompt
            + "<|im_start|>assistant\n" # assistant prompt start
    
    ```

  - v0.9
    文本格式如下：
    ```
    <|im_start|>system
    你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。<|im_end|>
    <|im_start|>user
    将下面的日文文本翻译成中文：日文第一行
    日文第二行
    日文第三行
    ...
    日文第n行<|im_end|>
    <|im_start|>assistant
    
    ```
    使用代码处理如下：
    ```python
    input_text_list = ['a', 'bb', 'ccc', ...] # 一系列上下文文本，每个元素代表一行的文本
    raw_text = "\n".join(input_text_list)
    prompt = "<|im_start|>system\n你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。<|im_end|>\n" \ # system prompt
            + "<|im_start|>user\n将下面的日文文本翻译成中文：" + raw_text + "<|im_end|>\n" \ # user prompt
            + "<|im_start|>assistant\n" # assistant prompt start
    ```

- prompt构建：

  - v0.8

    ```python
    input_text = "" # 要翻译的日文
    query = "将下面的日文文本翻译成中文：" + input_text
    prompt = "<reserved_106>" + query + "<reserved_107>"
    ```
    
  - v0.9

    ```python
    input_text = "" # 要翻译的日文
    query = "将下面的日文文本翻译成中文：" + input_text
    prompt = "<|im_start|>system\n你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。<|im_end|>\n<|im_start|>user\n" + query + "<|im_end|>\n<|im_start|>assistant\n"
    ```

- 推理与解码参数：

| 参数 | 值 |
| ---- | ---- |
| temperature | 0.1 |
| top p | 0.3 |
| do sample | True |
| beams number | 1 |
| repetition penalty | 1 |
| max new token | 512 |
| min new token | 1 |

**如出现退化（退化的例子可参见[#35](https://github.com/SakuraLLM/Sakura-13B-Galgame/issues/35)与[#36](https://github.com/SakuraLLM/Sakura-13B-Galgame/issues/36)），可增加`frequency_penalty`参数，并设置为大于0的某值，一般设置0.1~0.2即可。**

# 微调

模型微调框架参考[BELLE](https://github.com/LianjiaTech/BELLE)或[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，prompt构造参考推理部分。

# 相关项目

- [轻小说机翻机器人](https://books.fishhawk.top/)：轻小说翻译

- [LunaTranslator](https://github.com/HIllya51/LunaTranslator)：Galgame在线翻译

- [GalTransl](https://github.com/XD2333/GalTransl)：Galgame离线翻译，制作补丁

- [AiNiee](https://github.com/NEKOparapa/AiNiee-chatgpt)：RPG游戏翻译

# 致谢

- [CjangCjengh](https://github.com/CjangCjengh)

- [ryank231231](https://github.com/ryank231231)

- [KurikoMoe](https://github.com/kurikomoe)

- [FishHawk](https://github.com/FishHawk)

- [K024](https://github.com/K024)

- [minaduki-sora](https://github.com/minaduki-sora)

- [Kimagure7](https://github.com/Kimagure7)

- [YYF233333](https://github.com/YYF233333)

- [Isotr0py](https://github.com/Isotr0py)

- [XD2333](https://github.com/XD2333)

# Copyright Notice

v0.8版本模型的使用须遵守[Apache 2.0](https://github.com/baichuan-inc/Baichuan2/blob/main/LICENSE)、[《Baichuan 2 模型社区许可协议》](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/Baichuan%202%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf)和[CC BY-NC-SA 4.0协议](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)。

v0.9版本模型的使用须遵守[Qwen模型许可协议](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT)和[CC BY-NC-SA 4.0协议](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)。

v1.0版本模型的使用须遵守[CC BY-NC-SA 4.0协议](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)。

