import asyncio
from typing import *
from fastapi import Depends, Request
from pprint import pprint, pformat
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from transformers import GenerationConfig
from api.legacy.type import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse, OpenAIChatModelsResponse

from sse_starlette.sse import EventSourceResponse

from utils import state

import logging
import time

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/v1/chat",
)

async def get_output(data: OpenAIChatCompletionRequest) -> OpenAIChatCompletionResponse:
    logger.debug(f"Incoming request: \n{data.model_dump()}")
    generation_config = GenerationConfig(**data.compatible_with_backend())

    model = state.get_model()
    prompt, src_text = model.make_prompts_unstable(data.messages)
    generation_config.__dict__['src_text'] = src_text
    logger.info(f"translate: {prompt}")
    output = await model.completion_async(prompt, generation_config)
    # FIXME(kuriko): only for testing, remember to comment this out.
    # await asyncio.sleep(600)

    logger.info(f"answer: {output}")
    return OpenAIChatCompletionResponse(
        choices=[
            OpenAIChatCompletionResponse.Choice(
                finish_reason=output.finish_reason,
                index=0,
                message=OpenAIChatCompletionResponse.Choice.Message(
                    content=output.text,
                    role="assistant"
                )
            )
        ],
        created=int(time.time()),
        id="114514",
        model=f"{model.cfg.model_name}-{model.cfg.model_version}-{model.cfg.model_quant}",
        object="chat.completion",
        usage=OpenAIChatCompletionResponse.Usage(
            completion_tokens=output.new_token,
            prompt_tokens=output.prompt_token,
            total_tokens=output.new_token + output.prompt_token
        )
    )

@router.get("/model_info")
async def get_model_info():
    cfg = state.get_model().cfg
    metadata = {
        "model_name": cfg.model_name,
        "model_version": cfg.model_version,
        "model_quant": cfg.model_quant,
        "model_name_or_path": cfg.model_name_or_path
    }
    return JSONResponse(content=metadata)

@router.post("/completions")
async def completions(req: Request, data: OpenAIChatCompletionRequest):
    ret = await get_output(data)
    json_compatible_item_data = jsonable_encoder(ret)
    return JSONResponse(content=json_compatible_item_data)

@router.post("/stream/completions")
async def completions(req: Request, data: OpenAIChatCompletionRequest):
    async def generator():
        try:
            ret: OpenAIChatCompletionResponse = await get_output(data)
            yield dict(data=ret.model_dump_json())
        except asyncio.CancelledError as e:
            logger.warning(f"Disconnected from client (via refresh/close) {req.client})")
            raise e

    return EventSourceResponse(
        generator(),
        ping=15,
        media_type = "text/event-stream",
    )
