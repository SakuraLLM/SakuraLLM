from typing import *
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from api.legacy.type import OpenAIChatModelsResponse
from utils import state
import logging

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/v1",
)

@router.get("/models")
async def get_model_info():
    cfg = state.get_model().cfg
    # metadata = {
    #     "model_name": cfg.model_name,
    #     "model_version": cfg.model_version,
    #     "model_quant": cfg.model_quant,
    #     "model_name_or_path": cfg.model_name_or_path
    # }
    models_list = OpenAIChatModelsResponse(
        object="list",
        data=[OpenAIChatModelsResponse.OpenAIChatModel(
            id=f"{cfg.model_name}-{cfg.model_version}-{cfg.model_quant}",
            created="114514",
            object='model',
            owned_by='SakuraLLM',
            model_name=cfg.model_name,
            model_version=cfg.model_version,
            model_quant=cfg.model_quant,
            model_name_or_path=cfg.model_name_or_path
        )]
    )
    models_list = jsonable_encoder(models_list)
    return JSONResponse(content=models_list)

