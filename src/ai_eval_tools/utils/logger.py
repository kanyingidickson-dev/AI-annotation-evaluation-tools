import logging
import sys
from typing import Optional

def setup_logger(name: str = "ai_eval_tools", level: int = logging.INFO) -> logging.Logger:
    """
    Configures a professional logger with consistent formatting.
    
    Args:
        name: The name of the logger.
        level: The logging level (default: INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logger()
