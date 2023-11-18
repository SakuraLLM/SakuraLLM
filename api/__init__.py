from fastapi import Request
import logging

logger = logging.getLogger(__name__)


def log_request(req: Request):
    logger.info(f"Incoming request api: {req.url}")
    logger.info(f"client: {req.client}")
    return