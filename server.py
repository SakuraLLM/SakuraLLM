import asyncio
import coloredlogs
import logging
import os
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from pprint import pprint, pformat
import random
from dacite import from_dict
from hypercorn import Config

# import uvicorn
from hypercorn.asyncio import serve
from fastapi import FastAPI, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware

from api import log_request
from api.auth import get_auth_username

from utils import *
from utils import model as M
from utils import state
from utils.state import ServerConfig


dependencies = [
    Depends(log_request),
]

# parse config
parser = ArgumentParser()
# server config
parser.add_argument("--listen", type=str, default="127.0.0.1:5000")
parser.add_argument("--auth", type=str, default=None,
                    help="user:pass, user & pass should not contain ':'")
parser.add_argument("--no-auth", action="store_true",
                    help="force disable auth")

# log
parser.add_argument("-l", "--log", dest="logLevel", choices=[
                    'trace', 'debug', 'info', 'warning', 'error', 'critical'], default="info", help="Set the logging level")

# model config
parser.add_argument("--model_name_or_path", type=str,
                    default="SakuraLLM/Sakura-13B-LNovel-v0.8", help="model huggingface id or local path.")
parser.add_argument("--use_gptq_model", action="store_true",
                    help="whether your model is gptq quantized.")
parser.add_argument("--model_version", type=str, default="0.8",
                    help="model version written on huggingface readme, now we have ['0.1', '0.4', '0.5', '0.7', '0.8']")
parser.add_argument("--trust_remote_code", action="store_true",
                    help="whether to trust remote code.")

parser.add_argument("--llama", action="store_true",
                    help="whether your model is llama family.")

parser.add_argument("--llama_cpp", action="store_true", help="whether to use llama.cpp.")
parser.add_argument("--use_gpu", action="store_true", help="whether to use gpu when using llama.cpp.")
parser.add_argument("--n_gpu_layers", type=int, default=0, help="layers cnt when using gpu in llama.cpp")
parser.add_argument("--dry_run", action="store_true", help="whether to use gpu when using llama.cpp.")
args = parser.parse_args()


coloredlogs.install(level=args.logLevel.upper())
logger = logging.getLogger(__name__)
logger.debug(f"Current Log Level: {args.logLevel}")


addr = args.listen.split(":")

ServerConfig.address = addr[0]
ServerConfig.port = int(addr[1])

# Hidden trick to disable auth, useful when you use docker-compose
if args.auth == ":":
    args.auth = None
    args.no_auth = True

auth = [None, None]
if args.no_auth:
    logger.warning("Auth is disabled!")
else:
    if not args.auth:
        # Generate random auth credentials
        auth = f"sakura:{random.randint(114514, 19194545)}"
        logger.warning(f"Using random auth credentials. {auth}")

    auth = args.auth.split(":")
    # Insert http auth check
    dependencies.append(Depends(get_auth_username))

ServerConfig.username = auth[0]
ServerConfig.password = auth[1]

app = FastAPI(dependencies=dependencies)

from api.legacy import router as legacy_router
app.include_router(legacy_router)

from api.core import router as core_router
app.include_router(core_router)

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
    logger.info(f"Current server config: {ServerConfig.show()}")

    # build cfg from args
    cfg = from_dict(data_class=M.SakuraModelConfig, data=args.__dict__)

    logger.info(f"Current model config: {cfg}")
    state.init_model(cfg)
    state.get_model().check_model_by_magic()

    logger.info(
        f"Server will run at http://{ServerConfig.address}:{ServerConfig.port}, preparing...")

    # disable multiprocessing, since LLM model is not thread safe
    if False:  # use uvicorn
        uvicorn.run("server:app",
                    host=ServerConfig.address,
                    port=ServerConfig.port,
                    log_level=args.logLevel,
                    workers=1
                    )
    else:  # use hypercorn
        config = Config()
        binding = f"{ServerConfig.address}:{ServerConfig.port}"
        logger.debug(f"hypercorn binding: {binding}")
        config.bind= [binding,]
        config.loglevel = args.logLevel
        config.debug = args.logLevel == "debug"

        asyncio.run(serve(app, config))
