## Prerequisite 

0. A stable internet connection to huggingface and dockerhub. the scripts need to download the latest tokenization_baichuan.py and other things. So please ensure that you can connect to these sites.

1. Install `nvidia-container-runtime` according to [Nvidia Website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

    1.1 Remember `sudo systemctl restart docker` after `nvidia-container-runtime` installed. 

    1.2 You can check whether gpu is supported or not by the following command

    ```shell
    docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu20.04 nvidia-smi
    ```

2. Download model from [sakuraumi/Sakura-13B-Galgame](https://huggingface.co/sakuraumi/Sakura-13B-Galgame) and put it into models folder

## Hardware and Environments

As for now, [轻小说机翻机器人](https://books.fishhawk.top/) is using [Sakura-13B-LNovel-v0.8-4bit](https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.8-4bit). 
We strongly recommend using 4bit version for the balance between GPU memory usage and translation speed.

4bit model on RTX 3090 is around 46 tokens/s, and consumes about 14GiB GPU memory when idle, and up to 18G when working.

wsl2 with wslg should work, but since I only have a 4060 laptop, there is no further test for now.


## Usage

### Server

Copy `compose.yaml.example` to `compose.yaml` and tweak the following settings in `compose.yaml` to ensure the safty.
- USERNAME
- PASSWORD

To start the service, simply use the command:
```shell
docker compose run server
# or docker compose up server
```

You can use the python script in `test/single.py` to test the connection and performance of your gpu.

> remember to change the username and password
```shell
python3 tests/single.py --auth sakura:itsmygo http://127.0.0.1:5000
```

> It seems docker version is a little slower than run scripts in host machine. If that applies, you can follow the setup instruction in Dockerfile to prepare your own environment.


### Translation

> TODO(kuriko)

> put file into models directory

```shell
docker compose run translate-epub {--data_path <EPUB> | --data_folder <EPUB folder>}  --output_folder <output>
docker compose run translate-novel --data_path <TXT> --output_path <TXT OUT> [--compare_text true|false]

# docker compose run translate-novel --data_path /models/a.txt --output_path /models/b.txt
```
