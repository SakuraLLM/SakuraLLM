from dataclasses import dataclass
from pydantic import BaseModel
from transformers import GenerationConfig
from types import *
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelTestCase:
    model_version: str
    generation_config: GenerationConfig
    test_input: str
    test_output: str


def get_test_case_by_model_version(model_version: str):
    if model_version == "0.8":
        return ModelTestCase(
            model_version=model_version,
            generation_config=GenerationConfig(
                num_beams=1,
                max_new_tokens=1024,
                min_new_tokens=1,
                do_sample=False,
                # do_sample cannot be used with these
                # top_k=40,
                # top_p=0.3,
                # temperature=0.1,
            ),
            test_input="やる気マンゴスキン",
            test_output="干劲Mangoskin",
        )
    else:
        logger.error(f"Invalid model version: {model_version}, please check the arguments")
        return None
