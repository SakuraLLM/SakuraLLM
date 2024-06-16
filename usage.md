# 各种推理引擎的使用说明

* [llama-cpp-python](#llama-cpp-python)
* [vllm](#vllm)
* [ollama](#ollama)

## llama-cpp-python
### 支持的模型类型
<!-- // 这部分写支持加载哪些模型，如果是非 SakuraLLM 支持的格式的话，哪些地方能找到第三方维护的转换后的格式。 -->
GGUF 量化模型。

### 常见问题
llama 模式启动后提示不支持 qwen2，报错为：unknown model architecture: 'qwen2'

相关 issue: https://github.com/SakuraLLM/SakuraLLM/issues/92 https://github.com/SakuraLLM/SakuraLLM/issues/71

原因，旧版本 llama_cpp_python 不支持 qwen2，需要按照 cuda 版本更新，如果没有按照 cuda 版本更新可能会不使用显卡。

```shell
# 卸载原有旧版本 llama-cpp-python
pip uninstall llama-cpp-python

# 安装新版，cuda-version 为 cu121、cu122、cu123、cu124，python 版本仅支持 3.10、3.11、3.12
pip install llama-cpp-python
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/

#假设为 cuda 12.1
pip install llama-cpp-python
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### 使用例子
以运行 [sakura-13b-lnovel-v0.9b-Q4_K_M.gguf](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.9b-GGUF) 模型为例：
```python
python server.py \
    --model_name_or_path ./models/sakura-13b-lnovel-v0.9b-Q4_K_M.gguf \
    --llama_cpp \
    --use_gpu \
    --model_version 0.9 \
    --trust_remote_code \
    --no-auth
```

### server.py 相关参数及说明
<!-- // 这部分写一下推理引擎用到了哪些参数，由于早期设计问题，server.py 没有对参数归类，导致混杂在一起了。 -->
```shell
# 通用参数
--model_name_or_path: GGUF 模型路径
--model_version: 0.9, 0.8
--no-auth: 强制禁用身份验证

# llama.cpp 特有参数
--llama_cpp: 启动 llama.cpp 推理引擎
--use_gpu: llama.cpp 使用 GPU 进行推理
--n_gpu_layers: 加载至 GPU 的模型层数
```

## vllm
vLLM 是一个快速且易于使用的 LLM 分布式推理和服务库。

优势：
1. 支持 PagedAttention。
2. 支持 tensor parallel 多 GPU 推理加速。
3. 支持 GPTQ, AWQ, SqueezeLLM, FP8 KV Cache 等量化方法。
4. 支持 NVIDIA GPU 和 AMD GPU（实验性，未测试）。
5. 与 HuggingFace Transformers 模型无缝集成。

劣势：
1. Sakura13B v0.9 与 v0.10 仅能运行全量模型（目前 SakuraLLM 未提供量化模型），显存用量高于 llama.cpp 与 ollama。
2. 仅支持 4bit 量化，且量化存在 bug，效果不如直接运行全量模型。
3. 由于存在部分依赖冲突，依赖安装相对繁琐。

### 支持的模型类型
<!-- // 这部分写支持加载哪些模型，如果是非 SakuraLLM 支持的格式的话，哪些地方能找到第三方维护的转换后的格式。 -->
Transformers 模型（包括 GPTQ 4bit 量化与 AWQ 量化）。

### 前置需求
<!-- // 这部分写除了 `requirements.txt` 中 package 依赖以外的依赖。 -->
1. 首先运行以下命令安装 vllm 库（必须先运行这一步，否则会有依赖冲突导致无法安装）：
   ```shell
   pip install vllm
   ```
2. 运行以下命令安装运行 vllm 后端的其余依赖库：
   ```shell
   pip install -r requirements.vllm.txt
   ```

### 使用例子
以运行 [SakuraLLM/Sakura-13B-LNovel-v0.9](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.9) 模型为例：
```python
python server.py \
    --model_name_or_path SakuraLLM/Sakura-13B-LNovel-v0.9 \
    --vllm \
    --model_version 0.9 \
    --trust_remote_code \
    --no-auth \
    --tensor_parallel_size 2 \
    --enforce_eager
```

### server.py 相关参数及说明
<!-- // 这部分写一下推理引擎用到了哪些参数，由于早期设计问题，server.py 没有对参数归类，导致混杂在一起了。 -->
```shell
# 通用参数
--model_name_or_path: transformers 模型 tag
--model_version: 0.9, 0.8
--use_gptq_model: 使用 GPTQ 量化模型
--use_awq_model: 使用 AWQ 量化模型
--trust_remote_code: 允许不安全代码
--no-auth: 强制禁用身份验证

# vllm 特有参数
--vllm: 启动 vllm 推理引擎
--enforce_eager: 启用 eager 模式，可减少显存用量
--tensor_parallel_size: tensor parallel 的规模，一般取可用的 GPU 数量
--gpu_memory_utilization: 0~1, vllm 推理引擎可用的每个 GPU 显存比例
```

## ollama

### 介绍
原项目地址：[ollama/ollama](https://github.com/ollama/ollama)

优势：
1. 安装运行简便，使用 docker 对模型进行管理。
2. 对于 kaggle 环境，从 [ollama library](https://ollama.com/library) 拉取模型速度快于 huggingface 仓库，可进行 API Server 快速部署。

劣势：
1. 使用 ollama 私有格式模型，需要对 gguf 和 PyTorch/Safetensors 格式模型进行转换。

### 支持的模型类型
<!-- // 这部分写 ollama 支持加载哪些模型，如果是非 SakuraLLM 支持的格式的话，哪些地方能找到第三方维护的转换后的格式。 -->
ollama 私有格式模型，可从 GGUF 和 PyTorch/Safetensors 格式模型进行转换，转换方法参见 [issue #49](https://github.com/SakuraLLM/Sakura-13B-Galgame/issues/49) 与 [ollama 文档](https://github.com/ollama/ollama/blob/main/docs/import.md)。

Sakura 相关模型地址（第三方维护）：
- [onekuma/sakura-13b-lnovel-v0.9b-q2_k](https://registry.ollama.ai/onekuma/sakura-13b-lnovel-v0.9b-q2_k/tags)

### 前置需求
<!-- // 这部分写除了 `requirements.txt` 中 package 依赖以外的依赖。 -->
1. 从[ollama 官网](https://registry.ollama.ai/download)下载并安装 ollama 程序，用于自动下载 ollama 模型。
2. 运行以下命令安装运行 ollama 后端的依赖库：
   ```shell
   pip install -r requirements.ollama.txt
   ```

### 使用例子
以运行 [onekuma](https://registry.ollama.ai/onekuma) 维护的 [sakura-13b-lnovel-v0.9b-q2_k](https://registry.ollama.ai/onekuma/sakura-13b-lnovel-v0.9b-q2_k/tags) 模型为例：
```python
python server.py \
    --model_name_or_path onekuma/sakura-13b-lnovel-v0.9b-q2_k \
    --ollama \
    --model_version 0.9 \
    --trust_remote_code \
    --no-auth
```

### server.py 相关参数及说明
<!-- // 这部分写一下 ollama 模型用到了哪些参数，由于早期设计问题，server.py 没有对参数归类，导致混杂在一起了。 -->
```shell
# 通用参数
--model_name_or_path: ollama 模型 tag (不支持文件路径)
--model_version: 0.9
--no-auth: 强制禁用身份验证

# ollama 特有参数
--ollama: 启动 ollama 推理引擎
```
