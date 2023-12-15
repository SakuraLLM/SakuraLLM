import os
import sys
# Fix for windows embedded environment
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import random
import asyncio
import coloredlogs
import logging
from argparse import ArgumentParser
from dacite import from_dict
from hypercorn import Config

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from api import log_request
from api.auth import get_auth_username

from utils import *
from utils import model as M
from utils import state
from utils.state import ServerConfig

from utils.cli import parse_args


dependencies = [
    Depends(log_request),
]

args = parse_args()

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
        args.auth = f"sakura:{random.randint(114514, 19194545)}"
        logger.warning(f"Using random auth credentials. {auth}")

    auth = args.auth.split(":")
    # Insert http auth check
    dependencies.append(Depends(get_auth_username))

ServerConfig.username = auth[0]
ServerConfig.password = auth[1]

app = FastAPI(dependencies=dependencies)

from api.legacy import router as legacy_router
app.include_router(legacy_router)

from api.openai.v1 import router as openai_router
app.include_router(openai_router)

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
        import uvicorn
        uvicorn.run("server:app",
                    host=ServerConfig.address,
                    port=ServerConfig.port,
                    log_level=args.logLevel,
                    workers=1
                    )
    else:  # use hypercorn
        from hypercorn.asyncio import serve
        config = Config()
        binding = f"{ServerConfig.address}:{ServerConfig.port}"
        logger.debug(f"hypercorn binding: {binding}")
        config.bind= [binding,]
        config.loglevel = args.logLevel
        config.debug = args.logLevel == "debug"

        asyncio.run(serve(app, config))
