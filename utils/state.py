from functools import lru_cache
from typing import *

from utils.model import SakuraModel

class ServerConfig:
    address: str = "127.0.0.1"
    port: int = 5000
    username: str|None = None
    password: str|None = None

    # command line arguments
    args: Any


sakura_model = None

def init_model(*args, **kwargs):
    global sakura_model
    sakura_model = SakuraModel(*args, **kwargs)

@lru_cache
def get_model():
    return sakura_model
