from fastapi import APIRouter, Depends

from api.auth import get_auth_username


router = APIRouter(
    prefix="/api/core/",
    dependencies=[
        Depends(get_auth_username)
    ]
)
