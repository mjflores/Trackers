# utils/logger.py
import logging
import sys
from typing import Optional


def get_logger(name: str,
               level: int = logging.INFO,
               fmt: Optional[str] = None) -> logging.Logger:

    logger = logging.getLogger(name)

    if logger.handlers:          # evita duplicar handlers
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
