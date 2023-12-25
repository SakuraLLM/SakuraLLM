<div align="center">
<h1>
  Sakura-13B-Galgame
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/sakuraumi/Sakura-13B-Galgame" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/models/sakuraumi/Sakura-13B-Galgame" target="_blank">ModelScope</a>
</p>

# 介绍

- 基于一系列开源大模型构建，在通用日文语料与轻小说/Galgame等领域的中日语料上进行继续预训练与微调，旨在提供性能接近GPT3.5且完全离线的Galgame/轻小说翻译大语言模型。

- 同时提供了运行模型的API后端，适配OpenAI API格式。

- 新建了[TG交流群](https://t.me/+QMDKZyO9GV1kNDA1)，欢迎交流讨论。

**如果使用模型翻译并发布，请在最显眼的位置标注机翻！！！！！开发者对于滥用本模型造成的一切后果不负任何责任。**

**对于模型翻译的人称代词问题（错用，乱加，主宾混淆，男女不分等）和上下文理解问题，如果有好的想法或建议，欢迎提issue！**

## 快速开始

详见[本仓库Wiki](https://github.com/SakuraLLM/Sakura-13B-Galgame/wiki).

## News

1. **[GalTransl](https://github.com/XD2333/GalTransl)已经支持Sakura API，可以通过本地部署API后端，在GalTransl中配置使用Sakura模型来翻译Galgame，制作内嵌式翻译补丁。**

   GalTransl是一个galgame自动化翻译工具，用于制作内嵌式翻译补丁。
  
2. 预览版v0.9.0pre3模型发布。该版本模型只是预览版本，目前可能仍存在问题。增加了约30亿字(~2.5B tokens)领域内日文语料数据进行继续预训练。

3. **Sakura API已经支持OpenAI格式，现在可以通过OpenAI库或者OpenAI API Reference上的请求形式与Server交互。**
一个使用OpenAI库与Sakura模型交互的例子详见[openai_example.py](https://github.com/SakuraLLM/Sakura-13B-Galgame/blob/main/tests/example_openai.py)。

4. 网站：[轻小说机翻机器人](https://books.fishhawk.top/)已接入Sakura模型(v0.8-4bit)，站内有大量模型翻译结果可供参考。你也可以自行部署模型并使用该网站生成机翻，目前已经支持v0.8与v0.9模型，且提供了llama.cpp一键包。
  
   轻小说机翻机器人网站是一个自动生成轻小说机翻并分享的网站。你可以浏览日文网络小说，或者上传Epub/Txt文件，并生成机翻。

5. [LunaTranslator](https://github.com/HIllya51/LunaTranslator)已经支持Sakura API，可以通过本地部署API后端，并在LunaTranslator中配置Sakura API来使用Sakura模型进行Galgame实时翻译。

   LunaTranslator是一个Galgame翻译工具，支持剪贴板、OCR、HOOK，支持40余种翻译引擎。

## 模型下载：
|   版本  | 全量模型 | GPTQ 8bit量化 | GPTQ 4bit量化 | GPTQ 3bit量化 | GGUF与量化 | AWQ量化
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| 20231026-v0.8 | 🤗 [Sakura-13B-LNovel-v0.8](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.8) | 🤗 [Sakura-13B-LNovel-v0_8-8bit](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0_8-8bit) | 🤗 [Sakura-13B-LNovel-v0_8-4bit](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0_8-4bit) | 🤗 [Sakura-13B-LNovel-v0_8-3bit](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0_8-3bit) | 🤗 [Sakura-13B-LNovel-v0_8-GGUF](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.8-GGUF) | 🤗 [Sakura-13B-LNovel-v0_8-AWQ](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0_8-AWQ) |
| 20231219-v0.9.0pre3 | 🤗 [Sakura-13B-LNovel-v0.9.0pre3](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.9.0pre3) | - | - | - | 🤗 [Sakura-13B-LNovel-v0.9.0pre3-GGUF](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.9.0pre3-GGUF) | - |

目前仍为实验版本，翻译质量在文风与流畅度上强于GPT-3.5，但词汇量逊于GPT-3.5. 个人使用推荐GPT4.

## TODO
- [x] 将`dev_server`分支合并到主分支，并将api格式改为openai like api格式。
- [ ] 支持多种后端至v0.9
    - [ ] `llama.cpp server`
    - [x] `llama-cpp-python`
    - [x] `autogptq`
    - [ ] `llm-sharp`
- [ ] 适配翻译工具
    - [ ] LunaTranslator(新API)
    - [x] GalTransl
    - [ ] BallonsTranslator
- [ ] 提供Python部署一键包
- [ ] 发布v0.9模型
- [ ] LoRA MoE模型测试
- [ ] ~7B模型测试
- [ ] ~30B模型测试

# 显存需求

下面的表格显示了使用不同量化和不同格式的模型时显存占用的大小。如果你的显卡显存不满足上述需求，可以尝试同时使用CPU与GPU进行推理。

- llama.cpp GGUF模型（使用v0.9.0pre1模型进行测试，v0.8模型与其类似）

|  模型量化类型  | 模型大小 | 推荐显存大小 |
|:-------:|:-------:|:-------:|
| fp16 | 26.3G | 超出游戏显卡显存范围 |
| Q8_0 | 14G | 24G |
| Q6_K | 11.4G | 20G |
| Q5_K_M | 10.1G | 16G |
| Q4_K_M | 8.8G | 16G |
| Q3_K_M | 7.2G | 16G |
| Q2_K | 6.1G | 12G |

- transformers autogptq模型（使用v0.8版本进行测试）

|  模型量化类型 | 推理显存(ctx约600) | 推理显存(ctx约1800) |
|:-------:|:-------:|:-------:|
| 全量 | 超出游戏显卡显存范围  | 超出游戏显卡显存范围  |
| 8bit | 21.1G | 23.4G |
| 4bit | 14.9G | 17.4G |
| 3bit | 13.7G | 15.5G |

# 日志

`20231125`: 上传第六版模型预览，改善数据集质量与格式，使用Qwen-14B-Chat模型进行继续预训练+微调，增加数据集。

`20231026`：上传第五版模型`sakura-13b-2epoch-3.8M-1025-v0.8`，改善数据集质量与格式，修复之前版本模型无法正确解析\n的问题，使用Baichuan2-13B-Chat模型进行微调。

`20231011`：上传第四版模型`sakura-14b-2epoch-4.4M-1003-v0.7`，改用QWen-14B-Chat模型进行微调，针对较长文本进行优化，增加数据集。

`20230918`：上传第三版模型的8bits量化版`sakura-13b-2epoch-2.6M-0917-v0.5-8bits`。

`20230917`：上传第三版模型`sakura-13b-2epoch-2.6M-0917-v0.5`，改用Baichuan2-13B-Chat模型进行微调，翻译质量有所提高。

`20230908`：上传第二版模型`sakura-13b-1epoch-2.6M-0903-v0.4`，使用Galgame和轻小说数据集进行微调，语法能力有所提高。感谢[CjangCjengh](https://github.com/CjangCjengh)大佬提供轻小说数据集。

`20230827`：上传第一版模型`sakura-13b-2epoch-260k-0826-v0.1`

# 模型详情

## 描述

- Finetuned by [SakuraUmi](https://github.com/pipixia244)
- Finetuned on [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
- Finetuned on [Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B)
- Languages: Chinese/Japanese

## 效果

- Galgame

  TBD
  
- 轻小说

  网站：[轻小说机翻机器人](https://books.fishhawk.top/)已接入Sakura模型(v0.8-4bit)，站内有大量模型翻译的轻小说可供参考。

- PPL/BLEU/Human

  TBD

# 推理

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

如出现退化，可增加`frequency_penalty`参数，并设置为大于0的某值，一般设置0.05~0.2即可。

# 微调

模型微调框架参考[BELLE](https://github.com/LianjiaTech/BELLE)或[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，prompt构造参考推理部分。

# 致谢

- [CjangCjengh](https://github.com/CjangCjengh)

- [ryank231231](https://github.com/ryank231231)

- [KurikoMoe](https://github.com/kurikomoe)

- [FishHawk](https://github.com/FishHawk)

- [K024](https://github.com/K024)

- [minaduki-sora](https://github.com/minaduki-sora)

- [Kimagure7](https://github.com/Kimagure7)

- [YYF233333](https://github.com/YYF233333)

# Copyright Notice

v0.8版本模型的使用须遵守[Apache 2.0](https://github.com/baichuan-inc/Baichuan2/blob/main/LICENSE)和[《Baichuan 2 模型社区许可协议》](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/Baichuan%202%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf)。

v0.9版本模型的使用须遵守[Qwen模型许可协议](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT)。
