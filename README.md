<div align="center">
<h1>
  Sakura-13B-Galgame
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/sakuraumi/Sakura-13B-Galgame" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/models/sakuraumi/Sakura-13B-Galgame" target="_blank">ModelScope</a>
</p>

# 介绍

基于LLaMA2-13B，OpenBuddy(v0.1-v0.4)和Baichuan2-13B(v0.5+)构建，在Galgame中日文本数据上进行微调，旨在提供性能接近GPT3.5且完全离线的Galgame/轻小说翻译大语言模型. 新建了[TG交流群](https://t.me/+sCYaCYEsd3ZkMTE1)，欢迎交流讨论。

模型下载：
|   版本  | 全量模型 | 8-bit量化 | 4-bit量化|
|:-------:|:-------:|:-------:|:-------:|
| 20230827-v0.1 | 🤗 [Sakura-13B-Galgame-v0.1](https://huggingface.co/sakuraumi/Sakura-13B-Galgame/tree/main/sakura_13b_model_v0.1) | - | - |
| 20230908-v0.4 | 🤗 [Sakura-13B-Galgame-v0.4](https://huggingface.co/sakuraumi/Sakura-13B-Galgame/tree/main/sakura_13b_model_v0.4) | - | - |
| 20230917-v0.5 | 🤗 [sakuraumi/Sakura-13B-Galgame默认模型](https://huggingface.co/sakuraumi/Sakura-13B-Galgame/tree/main/) | 🤗 [Sakura-13B-Galgame-v0.5-8bits](https://huggingface.co/sakuraumi/Sakura-13B-Galgame/tree/main/sakura_13b_model_v0.5_8bits) | [Sakura-13B-Galgame-v0.5-4bits](https://huggingface.co/sakuraumi/Sakura-13B-Galgame/tree/main/sakura_13b_model_v0.5_4bits_autogptq_40k) |
| 20231011-v0.7 | 🤗 [Sakura-14B-LNovel](https://huggingface.co/sakuraumi/Sakura-14B-LNovel) | - | - |

目前仍为实验版本，翻译质量较差. 

# 日志

`20231011`：上传第四版模型`sakura-14b-2epoch-4.4M-1003-v0.7`，改用QWen-14B-Chat模型进行微调，针对较长文本进行优化，增加数据集。

`20230918`：上传第三版模型的8bits量化版`sakura-13b-2epoch-2.6M-0917-v0.5-8bits`。

`20230917`：上传第三版模型`sakura-13b-2epoch-2.6M-0917-v0.5`，改用Baichuan2-13B-Chat模型进行微调，翻译质量有所提高。

`20230908`：上传第二版模型`sakura-13b-1epoch-2.6M-0903-v0.4`，使用Galgame和轻小说数据集进行微调，语法能力有所提高。感谢[CjangCjengh](https://github.com/CjangCjengh)大佬提供轻小说数据集。

`20230827`：上传第一版模型`sakura-13b-2epoch-260k-0826-v0.1`

# 模型详情

## 描述

### v0.1-v0.4

- Finetuned by [SakuraUmi](https://github.com/pipixia244)
- Finetuned on [Openbuddy-LLaMA2-13B](https://huggingface.co/OpenBuddy/openbuddy-llama2-13b-v8.1-fp16)
- Base model: [LLaMA2-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
- Languages: Chinese/Japanese

### v0.5

- Finetuned by [SakuraUmi](https://github.com/pipixia244)
- Finetuned on [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
- Base model: [Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)
- Languages: Chinese/Japanese

### v0.7

- Finetuned by [SakuraUmi](https://github.com/pipixia244)
- Finetuned on [Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B)
- Base model: [Qwen-14B](https://huggingface.co/Qwen/Qwen-14B)
- Languages: Chinese/Japanese

## 版本

### v0.7

数据集：约1M Galgame中日文本 + 约3.4M 轻小说中日文本

微调方式：全参数

微调epoch数：2

参数量：14B

### v0.5

数据集：约260k Galgame中日文本 + 约2.3M 轻小说中日文本

微调方式：全参数

微调epoch数：2

参数量：13B

### v0.4

数据集：约260k Galgame中日文本 + 约2.3M 轻小说中日文本

微调方式：全参数

微调epoch数：1

参数量：13B

### v0.1

数据集：约260k Galgame中日文本

微调方式：全参数

微调epoch数：2

参数量：13B

## 效果

- Galgame

|  原文   |  Ours(v0.5)  |  Ours(v0.4)  |  Ours(v0.1)  | ChatGPT(GPT-3.5) |
|  ----  | ---- | ---- | ---- | ---- |
| 「女の子の一人暮らしって、やっぱ一階は防范的に危ないのかな～？ お父さんには、一階はやめとけ～って言われててね？」 | 「女生一个人住，一楼在防范上果然很危险吗～？我爸爸叫我不要住一楼哦。」 | 「女孩子一个人住，果然还是不太安全吧～？爸爸说过，不要住一楼～」 |  「一个女孩子住在一楼，还是有点不太安全吧？爸爸说让我不要住在一楼」 | "一个女孩子独自一人住，大概一楼会不安全吧～？爸爸对我说过，一楼最好不要住～" |
| 「助けて、誰か助けてって思いながら、ただただ泣いてたんです……」 | 「我一边想着谁来救救我，一边不停地哭……」 |「我一边想着有没有人能救救我，一边哭哭啼啼……」 |  「我一边祈祷着，祈祷着有人能来救救我们，一边不停地哭泣……」| 「帮帮我，我一边想着有人帮助我，一边只是哭泣着……」 |
| 「そうだよ。これが太一の普通の顔だって。でも、ちょっと不気味だから、わたしみたいにニッコリ笑ってみて？」 | 「对啊。这就是太一平常的表情。不过，这样有点毛骨悚然，所以试着像我这样笑吧？」 |「对啊，这就是太一的普通表情。不过，感觉有点诡异，你像我一样笑咪咪地试试看？」 |「是啊。这就是太一的普通表情。但是，因为有点吓人，所以你也试着像我一样微笑一下吧？」 | “是的呢，这就是太一的平常表情哦。不过，有点怪异，所以像我这样放个甜甜的笑容试试看？” |
| 「そういうヒトの感情は、発情期を迎えてもいないネコには難しい」 | 「对于还没到发情期的猫来说，人类的这种感情实在是有点难以理解」| 「这种人类的感情，对还没进入发情期的猫来说太难懂了。」 |「这种人类的感情，对还没有迎来发情期的猫来说太难懂了」 | 这种人类的情感对于尚未进入发情期的猫来说是复杂的。 |
| 「朝になって、病院に行くまで。ずっと、ずーっとそばに居てくれて……」 | 「从早上到去医院的这段时间，一直、一直陪在我身边……」 | 「一直陪伴着我，直到早上去医院为止……」  |「一直陪我到早上去医院。一直，一直陪在我身边……」 | "直到早晨去医院为止。一直，一直都在我身旁……" |
| 「それ以外は、自由に過ごしているため、各自が好きにできる、とても平和な部活だった……。」 | 「除此之外，由于可以自由活动，大家都能随心所欲，所以是非常和平的社团活动……」 |「除此之外，我们都可以自由活动，每个人都能随心所欲，是个非常和平的社团……」 | 「除此之外，社团活动都是自由参加的，每个人都可以按自己的意愿去做自己想做的事情，所以社团活动也是非常和平的……」 | 「除此以外，因为大家都自由自在地度过时间，是个每个人都能按自己喜好随意参与的非常和平的社团活动......。」|
| 「そーだそーだ。せっかくお店休みにして遊びに来たのに」 | 「对呀对呀，难得我们关店跑出来玩耶。」 | 「没错没错，难得店里放假，我们才来玩的。」 |「是啊是啊，难得休息一天，我还想出来玩一下呢」| "是啊是啊。本来店铺难得休息，特地过来玩的呢。" |
| 伝えなければ、伝わらない。きっと、大事なことであるほど。 | 不表达就传达不了。越是重要的事情，越是如此。 | 不说出来就不会知道。越是重要的事情，就越是不能不说。 | -- | 如果不传达，就不会传达。毫无疑问，对于重要的事情来说是如此。 |
| が、ハチロクを手伝うことでそれが果たせるというのなら、仕事がどれほど増えようと、決して苦とは感じない。 | 不过，如果帮忙八六能让我实现这个愿望，无论工作增加多少，我也绝不会觉得痛苦。 | 不过，如果帮助八六就能实现这个愿望，不管工作多么繁重，我都不会觉得辛苦。 | -- | 如果通过帮助八六实现这一目标，无论工作增加多少，我绝不会感到苦恼。 |

- Novel

  使用[该仓库](https://github.com/FishHawk/sakura-test)的测试文本，仓库内提供了测试代码，测试文本以及v0.5版本的测试结果。使用该仓库代码在v0.7模型上对测试文本进行测试的结果已上传到[sakuraumi/Sakura-13B-Galgame-Archived](https://huggingface.co/sakuraumi/Sakura-13B-Galgame-Archived/blob/main/text.sa-packed)。

# 推理

- Galgame翻译的prompt构建：

  - v0.1

    ```python
    input_text = "" # 用户输入
    query = "将下面的日文文本翻译成中文：" + input_text
    prompt = "Human: \n" + query + "\n\nAssistant: \n"
    ```
    
  - v0.4

    ```python
    input_text = "" # 用户输入
    query = "将下面的日文文本翻译成中文：" + input_text
    prompt = "User: " + query + "\nAssistant: "
    ```

  - v0.5

    ```python
    input_text = "" # 用户输入
    query = "将下面的日文文本翻译成中文：" + input_text
    prompt = "<reserved_106>" + query + "<reserved_107>"
    ```
    
  - v0.7
    参考Qwen-14B-Chat的prompt构造方式：[这里](https://huggingface.co/Qwen/Qwen-14B-Chat/blob/5188dfeb4ff175705aa3a84ef9d616c70dea029b/qwen_generation_utils.py#L119)和[这里](https://github.com/hiyouga/LLaMA-Efficient-Tuning/blob/5310e4d1829f36619c8f224d09ec15eeaf7a4877/src/llmtuner/extras/template.py#L546)


- 推理与解码参数：

| 参数 | 值 |
| ---- | ---- |
| temperature | 1 |
| top p | 0.5 |
| do sample | True |
| beams number | 1 |
| repetition penalty | 1 |
| max new token | 512 |
| min new token | 1 |

- 量化：

根据transformers文档中给出的AutoGPTQ量化教程自行量化，或使用我们已经量化好的模型。

# 微调

流程与LLaMA2(v0.1-v0.4)/Baichuan2(v0.5+)/Qwen14B(v0.7)一致，prompt构造参考推理部分

# 后续工作

1. 优化数据集，主要优化数据集质量
2. 支持上下文理解
3. 支持指定专有名词

# 致谢

- [CjangCjengh](https://github.com/CjangCjengh)

- [ryank231231](https://github.com/ryank231231)

- 三日月クリ

- [K024](https://github.com/K024)

- [minaduki-sora](https://github.com/minaduki-sora)

- [Kimagure7](https://github.com/Kimagure7)

- [YYF233333](https://github.com/YYF233333)

# Copyright Notice

This model is built upon Meta's LLaMA series of models and is subject to Meta's licensing agreement.

This model is intended for use only by individuals who have obtained approval from Meta and are eligible to download LLaMA.

If you have not obtained approval from Meta, you must visit the https://ai.meta.com/llama/ page, read and agree to the model's licensing agreement, submit an application, and wait for approval from Meta before downloading the model from this page.
