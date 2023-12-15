from typing import *
from pydantic import BaseModel

FloatOrInt = Union[float, int]

class GenerateRequest(BaseModel):
    """Generate request class used in legacy api."""
    
    prompt: str
    # auto_max_new_tokens: bool = False
    # max_tokens_second: int
    # # Generation params. If 'preset' is set to different than 'None' the values
    # # in presets/preset-name.yaml are used instead of the individual numbers.
    # preset: str | None = None
    max_new_tokens: int
    do_sample: bool
    temperature: float | int
    top_p: float | int
    repetition_penalty: float | int
    num_beams: int
    # typical_p: float | int = 1
    # epsilon_cutoff: float | int = 0  # In units of 1e-4
    # eta_cutoff: float | int = 0  # In units of 1e-4
    # tfs: float | int = 1
    # top_a: float | int  = 0
    # presence_penalty: float | int = 0
    # frequency_penalty: float | int = 0
    # repetition_penalty_range: float | int
    top_k: int
    # min_length: float | int = 0
    # no_repeat_ngram_size: float | int = 0
    # penalty_alpha: float | int = 0
    # length_penalty: float | int = 1
    # early_stopping: bool = False
    # mirostat_mode: float | int = 0
    # mirostat_tau: float | int = 5
    # mirostat_eta: float | int = 0.1
    # grammar_string: str = ""
    # guidance_scale: float | int = 1
    # negative_prompt: str = ""
    seed: int
    # add_bos_token: bool = True
    # truncation_length: float | int = 2048
    # ban_eos_token: bool = False
    # custom_token_bans: str = ""
    # skip_special_tokens: bool = True
    # stopping_strings: List[Any] = []

    # Allow extra parameters
    class Config:
        extra = "allow"


class OpenAIChatCompletionRequest(BaseModel):
    """Generate request class used in openai api."""
    
    # OpenAI param
    messages: list[dict[str, str]]
    model: str = ""
    frequency_penalty: float | int = 0.0
    max_tokens: int
    seed: int = -1
    temperature: float | int
    top_p: float | int
    
    stop: list[list[str]] = None        # Only transformers backend support
    stream: bool = False                # NotImplement

    # presence_penalty: float | int     # extra param
    n: int = 1                          # extra param

    # logit_bias: dict                  # won't support
    # response_format: dict[str, str]   # won't support
    # tools: list                       # won't support
    # tool_choice: str | dict           # won't support
    # user: str                         # won't support

    # SakuraLLM param
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float | int = 1.0

    # Allow extra parameters
    class Config:
        extra = "allow"

    def compatible_with_backend(self):
        return {
            "messages": self.messages,
            "model": self.model,
            "n": self.n,
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": 40,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty
        }
    

class GenerateResponse(BaseModel):
    """Generate response class used in legacy api."""
    
    class Result(BaseModel):
        new_token: int
        text: str

    results: List[Result]


class OpenAIChatCompletionResponse(BaseModel):
    """Generate response class used in openai api."""
    
    class Choice(BaseModel):
        class Message(BaseModel):
            content: str
            role: str
        finish_reason: str
        index: int
        message: Message
    class Usage(BaseModel):
        completion_tokens: int
        prompt_tokens: int
        total_tokens: int
        
    choices: List[Choice]
    created: int
    id: str
    model: str
    object: str
    usage: Usage

'''
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
        "role": "assistant"
      }
    }
  ],
  "created": 1677664795,
  "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
  "model": "gpt-3.5-turbo-0613",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 17,
    "prompt_tokens": 57,
    "total_tokens": 74
  }
}
'''