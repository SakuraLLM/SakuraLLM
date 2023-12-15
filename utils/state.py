from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import *

from utils.model import SakuraModel

class ServerConfig:
    address: str = "127.0.0.1"
    port: int = 5000
    username: str|None = None
    password: str|None = None

    @classmethod
    def show(cls) -> str:
        return f"Server(listen: {cls.address}:{cls.port}, auth: {cls.username}:{cls.password})"

sakura_model = None

g_pool = ThreadPoolExecutor()

def init_model(*args, **kwargs):
    global sakura_model
    sakura_model = SakuraModel(*args, **kwargs)

@lru_cache
def get_model():
    return sakura_model
