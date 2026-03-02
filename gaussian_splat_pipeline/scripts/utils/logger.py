"""
Dual console + file logger for the pipeline.
All stage scripts import this first.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_file: Path = None, level=logging.INFO) -> logging.Logger:
    """Create a logger that writes to both console and optional log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_pipeline_logger(output_dir: Path, stage_name: str) -> logging.Logger:
    """Standard logger for pipeline stages."""
    log_file = Path(output_dir) / "logs" / "pipeline.log"
    return setup_logger(stage_name, log_file)
