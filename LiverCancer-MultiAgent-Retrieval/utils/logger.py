"""
Basic logging utilities.
"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a standard logger.
    
    Args:
        name: Name of the logger.
        log_file: Optional path to save logs.
        level: Logging level.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    if log_file is not None:
        log_path = Path(log_file)
        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
