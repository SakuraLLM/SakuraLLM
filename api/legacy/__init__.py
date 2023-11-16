from typing import *
from fastapi import Depends
from pprint import pprint, pformat
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from transformers import GenerationConfig
from fastapi.security import HTTPBasic
from api.legacy.type import GenerateRequest, GenerateResponse

from utils import state
from api.auth import get_auth_username

import logging
logger = logging.getLogger(__name__)



router = APIRouter(
    prefix="/api/v1",
)


@router.post("/generate")
def completions(data: GenerateRequest):
    # Mixin the required parameter with optional ones in extra.
    logger.debug(f"Incoming request: \n{data.model_dump()}")

    generation_config = GenerationConfig(**data.model_dump())

    logger.info(f"translate: {data.prompt}")
    output = state.get_model().completion(data.prompt, generation_config)
    logger.info(f"answer: {output}")

    ret: GenerateResponse = GenerateResponse(
        results=[
            GenerateResponse.Result(
                new_token =  output.new_token,
                text = output.text,
            )
        ]
    )

    json_compatible_item_data = jsonable_encoder(ret)
    return JSONResponse(content=json_compatible_item_data)
