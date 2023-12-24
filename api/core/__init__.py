from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.auth import get_auth_username
from utils.state import get_model


router = APIRouter(prefix="/api/core")

@router.get("/version")
def version():
    cfg = get_model().get_cfg()

    config = {
        "name": cfg.model_name,
        "version": cfg.model_version,
        "quant": cfg.model_quant,
    }
    return JSONResponse(content=config)
