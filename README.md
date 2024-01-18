<div align="center">
<h1>
  Sakura-13B-Galgame
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/sakuraumi/Sakura-13B-Galgame" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/models/sakuraumi/Sakura-13B-Galgame" target="_blank">ModelScope</a>
</p>

# 介绍

- 基于一系列开源大模型构建，在通用日文语料与轻小说/Galgame等领域的中日语料上进行继续预训练与微调，旨在提供性能接近 GPT3.5 且完全离线的Galgame/轻小说翻译大语言模型。

- 目前仍为实验版本，v0.9 版本模型在文风、流畅度与准确性上均强于 GPT-3.5，但词汇量略逊于 GPT-3.5（主观评价）.

- 同时提供了运行模型的 API 后端，适配 OpenAI API 格式。

- 新建了[TG交流群](https://t.me/+QMDKZyO9GV1kNDA1)，欢迎交流讨论。

**如果使用模型翻译并发布，请在最显眼的位置标注机翻！！！！！开发者对于滥用本模型造成的一切后果不负任何责任。**

**目前模型翻译仍存在一些人称代词问题（错用，乱加，主宾混淆，男女不分等）和上下文理解问题，如果有好的想法或建议，欢迎提 Issue & PR！**

## 快速开始

### 教程：

详见[本仓库Wiki](https://github.com/SakuraLLM/Sakura-13B-Galgame/wiki).

### 模型下载：

|   版本  | Transformers 模型 | GGUF 量化模型 | GPTQ 8bit 量化 | GPTQ 4bit 量化 | GPTQ 3bit 量化 | AWQ 量化
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| 20231026-v0.8 | 🤗 [Sakura-13B-LNovel-v0.8](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.8) | 🤗 [Sakura-13B-LNovel-v0_8-GGUF](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.8-GGUF) | 🤗 [Sakura-13B-LNovel-v0_8-8bit](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0_8-8bit) | 🤗 [Sakura-13B-LNovel-v0_8-4bit](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0_8-4bit) | 🤗 [Sakura-13B-LNovel-v0_8-3bit](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0_8-3bit) | 🤗 [Sakura-13B-LNovel-v0_8-AWQ](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0_8-AWQ) |
| 20240111-v0.9 | 🤗 [Sakura-13B-LNovel-v0.9](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.9) | 🤗 [Sakura-13B-LNovel-v0.9-GGUF](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.9-GGUF) | - | - | - | - |

## News

1.  **更新了 0.9 的第一个正式版模型`v0.9`。修复若干bug，清洗并增加了预训练与微调的数据量。更推荐使用正式版模型，它会比之前的 pre 版本更加稳定，质量更高。**

1. **增加了 vllm 模型后端的支持，详见**[#40](https://github.com/SakuraLLM/Sakura-13B-Galgame/pull/40)

1.  [感谢 fkiliver](https://github.com/fkiliver) 提供了用来翻译 Unity 引擎游戏的工具 [SakuraTranslator](https://github.com/fkiliver/SakuraTranslator) 与翻译 RPGMaker 引擎游戏的工具[RPGMaker_LLaMA_Translator](https://github.com/fkiliver/RPGMaker_LLaMA_Translator)。

1.  感谢 [Isotr0py](https://github.com/Isotr0py) 提供运行模型的 Notebook 仓库 [SakuraLLM-Notebooks](https://github.com/Isotr0py/SakuraLLM-Notebooks)，可在 [Colab](https://colab.research.google.com/) (免费 T4\*1) 与 [Kaggle](https://www.kaggle.com/) (免费 P100\*1 或 T4\*2) 平台使用。**已经更新 Kaggle 平台的[使用教程](https://github.com/SakuraLLM/Sakura-13B-Galgame/wiki/%E7%99%BD%E5%AB%96Kaggle%E5%B9%B3%E5%8F%B0%E9%83%A8%E7%BD%B2%E6%95%99%E7%A8%8B)，可以白嫖一定时间的 T4\*2。**

1.  **Sakura API 已经支持 OpenAI 格式，现在可以通过 OpenAI 库或者 OpenAI API Reference 上的请求形式与 Server 交互。**
一个使用 OpenAI 库与 Sakura 模型交互的例子详见 [openai_example.py](https://github.com/SakuraLLM/Sakura-13B-Galgame/blob/main/tests/example_openai.py)。

## 已经接入模型的工具

1. 网站：[轻小说机翻机器人](https://books.fishhawk.top/)已接入Sakura模型(v0.8-4bit)，站内有大量模型翻译结果可供参考。你也可以自行部署模型并使用该网站生成机翻，目前已经支持 v0.8 与 v0.9 模型，且提供了 llama.cpp 一键包。

   轻小说机翻机器人网站是一个自动生成轻小说机翻并分享的网站。你可以浏览日文网络小说，或者上传 Epub/Txt 文件，并生成机翻。

1. [LunaTranslator](https://github.com/HIllya51/LunaTranslator) 已经支持 Sakura API，可以通过本地部署 API 后端，并在 LunaTranslator 中配置 Sakura API 来使用 Sakura 模型进行 Galgame 实时翻译。

   LunaTranslator 是一个 Galgame 翻译工具，支持剪贴板、OCR、HOOK，支持 40 余种翻译引擎。

1. [GalTransl](https://github.com/XD2333/GalTransl) 已经支持 Sakura API，可以通过本地部署 API 后端，在 GalTransl 中配置使用 Sakura 模型来翻译 Galgame，制作内嵌式翻译补丁。

   GalTransl 是一个 Galgame 自动化翻译工具，用于制作内嵌式翻译补丁。一个使用 GalTransl 和 Sakura 模型翻译的[示例](https://www.ai2moe.org/files/file/2271-%E6%88%AF%E7%94%BBgaltranslsakuragpt35%E7%88%B1%E4%B9%8B%E5%90%BB3-sexy-gpt%E7%BF%BB%E8%AF%91%E8%A1%A5%E4%B8%81uploadee5-mb/)

## TODO
- [x] 将`dev_server`分支合并到主分支，并将 api 格式改为 openai like api 格式。
- [x] 支持多种后端至v0.9
    - [x] `llama.cpp server`
    - [x] `llama-cpp-python`
    - [x] `autogptq`
    - [x] `vllm`(同时支持 gptq 与 awq 模型)
- [ ] 适配翻译工具
    - [x] LunaTranslator(新API)
    - [x] GalTransl
    - [ ] BallonsTranslator
- [x] 提供 Python 部署一键包
- [x] 发布 v0.9 模型
- [ ] 发布 v0.9.1 模型
- [ ] ~7B 模型测试
- [ ] ~30B 模型测试
- [ ] LoRA MoE 模型测试

# 显存需求

下面的表格显示了使用不同量化和不同格式的模型时显存占用的大小。如果你的显卡显存不满足上述需求，可以尝试同时使用 CPU 与 GPU 进行推理。

- llama.cpp GGUF 模型（使用 v0.9.0pre1 模型进行测试，v0.8 模型与其类似）

|  模型量化类型  | 模型大小 | 推荐显存大小 |
|:-------:|:-------:|:-------:|
| fp16 | 26.3G | 超出游戏显卡显存范围 |
| Q8_0 | 14G | 24G |
| Q6_K | 11.4G | 20G |
| Q5_K_M | 10.1G | 16G |
| Q4_K_M | 8.8G | 16G |
| Q3_K_M | 7.2G | 16G |
| Q2_K | 6.1G | 12G |

- transformers autogptq 模型（使用 v0.8 版本进行测试）

|  模型量化类型 | 推理显存 (ctx 约 600) | 推理显存 (ctx 约 1800) |
|:-------:|:-------:|:-------:|
| 全量 | 超出游戏显卡显存范围  | 超出游戏显卡显存范围  |
| 8bit | 21.1G | 23.4G |
| 4bit | 14.9G | 17.4G |
| 3bit | 13.7G | 15.5G |

# 模型详情

## 描述

- Finetuned by [SakuraUmi](https://github.com/pipixia244)
- Finetuned on [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
- Finetuned on [Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)
- Finetuned on [Sakura-13B-Base-v0.9.0](https://huggingface.co/SakuraLLM/Sakura-13B-Base-v0.9.0)
- Languages: Chinese/Japanese

## 效果

- Galgame

  [一个例子](https://www.ai2moe.org/files/file/2271-%E6%88%AF%E7%94%BBgaltranslsakuragpt35%E7%88%B1%E4%B9%8B%E5%90%BB3-sexy-gpt%E7%BF%BB%E8%AF%91%E8%A1%A5%E4%B8%81uploadee5-mb/)

- 轻小说

  网站：[轻小说机翻机器人](https://books.fishhawk.top/)已接入 Sakura 模型 (v0.8-4bit)，站内有大量模型翻译的轻小说可供参考。

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

**如出现退化，可增加`frequency_penalty`参数，并设置为大于 0 的某值，一般设置 0.05~0.2 即可。**

# 微调

模型微调框架参考 [BELLE](https://github.com/LianjiaTech/BELLE) 或 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，prompt 构造参考推理部分。

# 相关项目

- [轻小说机翻机器人](https://books.fishhawk.top/)：轻小说翻译

- [LunaTranslator](https://github.com/HIllya51/LunaTranslator)：Galgame 在线翻译

- [GalTransl](https://github.com/XD2333/GalTransl)：Galgame 离线翻译，制作补丁

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

v0.8 版本模型的使用须遵守 [Apache 2.0](https://github.com/baichuan-inc/Baichuan2/blob/main/LICENSE) 和 [《Baichuan 2 模型社区许可协议》](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/Baichuan%202%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf)。

v0.9 版本模型的使用须遵守 [Qwen模型许可协议](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT)。
