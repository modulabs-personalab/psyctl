"""Simple logging setup."""

from __future__ import annotations

import logging
import sys
from typing import cast

from psyctl import config

# Add SUCCESS log level
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


class CustomLogger(logging.Logger):
    """Custom logger with success method."""

    def success(self, message, *args, **kwargs):
        """Log a success message."""
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log(SUCCESS_LEVEL, message, args, **kwargs)


def setup_logging():
    """Setup basic logging."""
    # Set custom logger class
    logging.setLoggerClass(CustomLogger)

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    # Console handler
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # File handler for DEBUG
    if config.LOG_LEVEL.upper() == "DEBUG":
        debug_file = config.OUTPUT_DIR / "debug.log"
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(debug_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    # Optional log file
    if config.LOG_FILE:
        config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str | None = None) -> CustomLogger:
    """Get logger instance."""
    logging.setLoggerClass(CustomLogger)
    logger = logging.getLogger(name or __name__)
    return cast("CustomLogger", logger)
