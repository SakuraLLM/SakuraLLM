import secrets
from typing import *
from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import status

from utils.state import ServerConfig as server_cfg


security = HTTPBasic()


def get_auth_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> str | None:
    if (correct_username := server_cfg.username) and (correct_password := server_cfg.password):
        correct_username_bytes = correct_username.encode('utf-8')
        correct_password_bytes = correct_password.encode('utf-8')

        current_username_bytes = credentials.username.encode("utf8")
        is_correct_username = secrets.compare_digest(
            current_username_bytes,
            correct_username_bytes
        )

        current_password_bytes = credentials.password.encode("utf8")
        is_correct_password = secrets.compare_digest(
            current_password_bytes,
            correct_password_bytes
        )

        if not (is_correct_username and is_correct_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username
    else:
        raise HTTPException(
            status_code=status.HTTP_418_IM_A_TEAPOT,
            detail="server started without auth is not recommended",
            headers={"WWW-Authenticate": "Basic"},
        )
