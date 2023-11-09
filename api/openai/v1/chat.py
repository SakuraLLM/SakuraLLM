from typing import *
from pydantic import BaseModel


class SystemMessage(BaseModel):
    content: str
    role: str = "system"

class UserMessage(BaseModel):
    content: str
    role: str = "user"

class UserMessage(BaseModel):
    content: str
    role: str = "user"


class Completion(BaseModel):
    messages: str
    model: str


    
