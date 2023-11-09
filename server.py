import os
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from pprint import pprint
import random

import uvicorn
from fastapi import FastAPI, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from api.auth import get_auth_username

from utils import *
from utils import model as M
from utils import state
from utils.state import ServerConfig

from api.legacy import router as legacy_router


import logging
import coloredlogs
coloredlogs.install()


# parse config
parser = ArgumentParser()
# server config
parser.add_argument("--listen", type=str, default="127.0.0.1:5000")
parser.add_argument("--auth", type=str, default=None, help="user:pass")
parser.add_argument("--no-auth", action="store_true", help="force disable auth")

# log
parser.add_argument("-l", "--log", dest="logLevel", choices=['trace', 'debug', 'info', 'warning', 'error', 'critical'], default="info", help="Set the logging level")

# model config
parser.add_argument("--model_name_or_path", type=str, default="SakuraLLM/Sakura-13B-LNovel-v0.8", help="model huggingface id or local path.")
parser.add_argument("--use_gptq_model", action="store_true", help="whether your model is gptq quantized.")
parser.add_argument("--model_version", type=str, default="0.8", help="model version written on huggingface readme, now we have ['0.1', '0.4', '0.5', '0.7', '0.8']")
parser.add_argument("--trust_remote_code", action="store_true", help="whether to trust remote code.")
parser.add_argument("--llama", action="store_true", help="whether your model is llama family.")
args = parser.parse_args()

logging.basicConfig(level=args.logLevel.upper())

ServerConfig.args = args

addr = args.listen.split(":")
ServerConfig.address = addr[0]
ServerConfig.port = int(addr[1])

dependencies=[]

# Hidden trick to disable auth, useful when you use docker-compose
if args.auth == ":":
    args.auth = None
    args.no_auth = True

if args.no_auth:
    logging.warn("Auth is disabled!")
    ServerConfig.username = None
    ServerConfig.password = None
else:
    if args.auth:
        auth = args.auth.split(":")
    else:
        # Generate random auth credentials
        auth = f"sakura:{random.randint(114514, 19194545)}"
        logging.warn(f"Using random auth credentials. {auth}")

    ServerConfig.username = auth[0]
    ServerConfig.password = auth[1]

    # Insert http auth check
    dependencies.append(Depends(get_auth_username))

app = FastAPI(dependencies=dependencies)

app.include_router(legacy_router)

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



if __name__ == "__main__":
    # copy k,v from args to cfg
    cfg = M.SakuraModelConfig()
    for k, v in ServerConfig.args.__dict__.items():
        cfg.__dict__[k] = v

    logging.info(f"Current config: {cfg.__dict__}")
    state.init_model(cfg)
    state.get_model().check_model_by_magic()

    logging.info(f"Server will run at http://{ServerConfig.address}:{ServerConfig.port}, preparing...")
    # disable multiprocessing, since LLM model is not thread safe
    uvicorn.run("server:app", host=ServerConfig.address, port=ServerConfig.port, log_level=args.logLevel, workers=1)
