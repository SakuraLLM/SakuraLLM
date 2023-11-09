from typing import *
from fastapi import Depends
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
    generation_config = GenerationConfig(**data.__dict__)

    logger.info(generation_config.__dict__)

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