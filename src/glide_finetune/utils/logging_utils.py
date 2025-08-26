"""Logging utilities for GLIDE finetune."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname:8s}{self.RESET}"
        return super().format(record)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Optional logger name. Defaults to 'glide_finetune'.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name or "glide_finetune")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        # Use colored formatter if stdout is a tty
        if sys.stdout.isatty():
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            handler.setFormatter(ColoredFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        else:
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    name: str | None = None,
    disable_colors: bool = False,
) -> logging.Logger:
    """Setup logging configuration with optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL or int)
        log_file: Optional path to log file
        name: Logger name (defaults to 'glide_finetune')
        disable_colors: Disable colored output even for TTY

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name or "glide_finetune")

    # Clear existing handlers
    logger.handlers.clear()

    # Set level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)

    # Use colored formatter if stdout is a tty and colors not disabled
    if sys.stdout.isatty() and not disable_colors:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_handler.setFormatter(ColoredFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    else:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        file_handler.setFormatter(logging.Formatter(file_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def set_log_level(level: str | int, name: str | None = None) -> None:
    """Set logging level for a logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL or int)
        name: Logger name (defaults to 'glide_finetune')
    """
    logger = logging.getLogger(name or "glide_finetune")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)


def disable_warnings() -> None:
    """Disable common warning messages from libraries."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Disable specific library warnings
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("torch.nn.parallel").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


def log_system_info(logger: logging.Logger | None = None) -> None:
    """Log system information for debugging.

    Args:
        logger: Logger instance (defaults to getting 'glide_finetune' logger)
    """
    import platform

    import torch

    if logger is None:
        logger = get_logger()

    logger.info("System Information:")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA: {torch.version.cuda}")
        logger.info(f"  GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"    GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        logger.info("  CUDA: Not available")
