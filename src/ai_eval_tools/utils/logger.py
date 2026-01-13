import logging
import sys
from typing import Optional

def setup_logger(name: str = "ai_eval_tools", level: int = logging.INFO, log_file: Optional[str] = "system.log") -> logging.Logger:
    """
    Configures a professional logger with consistent formatting and file output.
    
    Args:
        name: The name of the logger.
        level: The logging level (default: INFO).
        log_file: Path to log file for persistent audit trails.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent double logging if attached to root

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # File Handler (Audit)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

logger = setup_logger()
