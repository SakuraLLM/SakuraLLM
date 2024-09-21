import asyncio
import logging
import time
import json
from typing import *
from fastapi import Depends, Request
from pprint import pprint, pformat
from pydantic import BaseModel, ValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from transformers import GenerationConfig
from api.legacy.type import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse, OpenAIChatCompletionStreamResponse
from sse_starlette.sse import EventSourceResponse
from utils import state

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/v1/chat",
)

def get_output(data: OpenAIChatCompletionRequest) -> OpenAIChatCompletionResponse:
    logger.debug(f"Incoming request: \n{data.model_dump()}")

    model = state.get_model()
    generation_config = GenerationConfig(**data.compatible_with_backend(model.tokenizer))
    prompt = model.make_prompt_stable(data.messages)
    src_text = data.messages[-1]['content'].replace("将下面的日文文本翻译成中文：", "")
    generation_config.__dict__['src_text'] = src_text
    logger.info(f"translate: {prompt}")
    output = model.completion(prompt, generation_config)
    # FIXME(kuriko): only for testing, remember to comment this out.
    # await asyncio.sleep(600)

    logger.info(f"answer: {output}")

    prompt_tokens = output.prompt_token
    new_tokens = output.new_token

    total_tokens = None
    if prompt_tokens is not None:
        total_tokens = new_tokens + prompt_tokens

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
            completion_tokens=new_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )
    )

def get_stream_output(data: OpenAIChatCompletionRequest):
    logger.debug(f"Incoming request: \n{data.model_dump()}")
    model = state.get_model()
    generation_config = GenerationConfig(**data.compatible_with_backend(model.tokenizer))
    logger.info(f"current generation config: \n{pformat(generation_config.to_diff_dict())}")
    if data.messages is not None:
        logger.info(f"translate: {str(data.messages)}")
        src_text = data.messages[-1]['content'].replace("将下面的日文文本翻译成中文：", "")
        generation_config.__dict__['src_text'] = src_text
        stream_iterator = model.completion_stream(data.messages, generation_config)
    else:
        logger.info(f"prompt: {str(data.prompt)}")
        stream_iterator = model.completion_stream(data.prompt, generation_config)
    final_output = ""
    for idx, (output, finish_reason) in enumerate(stream_iterator):
        final_output += output
        try:
            if idx == 0:
                message = OpenAIChatCompletionStreamResponse.Choice.Message(role="assistant")
                yield message, OpenAIChatCompletionStreamResponse(
                    id="114514",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=f"{model.cfg.model_name}-{model.cfg.model_version}-{model.cfg.model_quant}",
                    system_fingerprint="fp_1919810",
                    choices=[OpenAIChatCompletionStreamResponse.Choice(
                        index=0,
                        delta=None,
                        logprobs=None,
                        finish_reason=finish_reason
                    )]
                )
            if finish_reason:
                message = OpenAIChatCompletionStreamResponse.Choice.Message()
            else:
                message = OpenAIChatCompletionStreamResponse.Choice.Message(content=output)
            yield message, OpenAIChatCompletionStreamResponse(
                id="114514",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=f"{model.cfg.model_name}-{model.cfg.model_version}-{model.cfg.model_quant}",
                system_fingerprint="fp_1919810",
                choices=[OpenAIChatCompletionStreamResponse.Choice(
                    index=0,
                    delta=None,
                    logprobs=None,
                    finish_reason=finish_reason
                )]
            )
        except ValidationError as e:
            print(e.json())

    logger.info(f"answer: {final_output}")

@router.get("/model_info")
def get_model_info():
    cfg = state.get_model().cfg
    metadata = {
        "model_name": cfg.model_name,
        "model_version": cfg.model_version,
        "model_quant": cfg.model_quant,
        "model_name_or_path": cfg.model_name_or_path
    }
    return JSONResponse(content=metadata)

@router.post("/completions")
def completions(req: Request, data: OpenAIChatCompletionRequest):
    if data.is_stream():
        def generator():
            try:
                for message, output in get_stream_output(data):
                    message_json = jsonable_encoder(message, exclude_none=True)
                    json_compatible_item_data = jsonable_encoder(output)
                    json_compatible_item_data['choices'][0]['delta'] = message_json
                    yield json.dumps(json_compatible_item_data, default=str, ensure_ascii=False)
            except asyncio.CancelledError as e:
                logger.warning(f"Disconnected from client (via refresh/close) {req.client})")
                raise e

        return EventSourceResponse(
            generator(),
            media_type = "text/event-stream",
        )
    else:
        ret = get_output(data)
        json_compatible_item_data = jsonable_encoder(ret)
        return JSONResponse(content=json_compatible_item_data)

# @router.post("/stream/completions")
# def completions(req: Request, data: OpenAIChatCompletionRequest):
#     def generator():
#         try:
#             if data.is_stream():
#                 for output in get_stream_output(data):
#                     json_compatible_item_data = jsonable_encoder(output)
#                     yield JSONResponse(content=json_compatible_item_data)
#             else:
#                 ret = get_output(data)
#                 json_compatible_item_data = jsonable_encoder(ret)
#                 yield JSONResponse(content=json_compatible_item_data)
#         except asyncio.CancelledError as e:
#             logger.warning(f"Disconnected from client (via refresh/close) {req.client})")
#             raise e

#     return EventSourceResponse(
#         generator(),
#         ping=15,
#         media_type = "text/event-stream",
#     )
