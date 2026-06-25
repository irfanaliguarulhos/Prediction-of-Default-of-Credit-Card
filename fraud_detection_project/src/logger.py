"""Shared logger support for the fraud detection pipeline.

This module prefers Loguru when installed, and falls back to Python's
standard logging if Loguru is unavailable.
"""

from pathlib import Path
import sys

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging

    logger = logging.getLogger("fraud_detection_project")
    LOGURU_AVAILABLE = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)


def setup_logging(config: dict):
    log_config = config.get('logging', {})
    log_file = log_config.get('file', 'output/logs/fraud_detection.log')
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
    rotation = log_config.get('rotation', '100 MB')
    retention = log_config.get('retention', '30 days')

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    if LOGURU_AVAILABLE:
        logger.remove()
        logger.add(sys.stdout, format=log_format, level=log_level)
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention
        )
    else:
        import logging

        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        for handler in list(logger.handlers):
            logger.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)
