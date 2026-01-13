"""
Logging Configuration - Structured logging setup

Provides:
- Centralized logging configuration
- File and console handlers
- Rotation and formatting
- Log level management
- Contextual logging
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from backend.constants import (
    LOG_DIR,
    DEFAULT_LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_FILE_MAX_BYTES,
    LOG_FILE_BACKUP_COUNT,
    APP_NAME
)


# ==================== LOGGER CONFIGURATION ====================

class ContextFilter(logging.Filter):
    """
    Add contextual information to log records.
    """
    
    def filter(self, record):
        """Add custom fields to log record."""
        # Add hostname
        import socket
        record.hostname = socket.gethostname()
        
        # Add process info
        import os
        record.process_id = os.getpid()
        
        # Add thread info
        import threading
        record.thread_name = threading.current_thread().name
        
        return True


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        """Format with colors for console."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for subsequent handlers
        record.levelname = levelname
        
        return formatted


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Setup structured logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Custom log file path
        console: Enable console logging
        file_logging: Enable file logging
        
    Returns:
        logging.Logger: Configured root logger
    """
    # Determine log level
    if level is None:
        level = DEFAULT_LOG_LEVEL
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create log directory if it doesn't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | '
            '%(funcName)s:%(lineno)d | %(message)s',
        datefmt=LOG_DATE_FORMAT
    )
    
    simple_formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )
    
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt=LOG_DATE_FORMAT
    )
    
    # Add context filter
    context_filter = ContextFilter()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(colored_formatter)
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_logging:
        if log_file is None:
            log_file = LOG_DIR / f"{APP_NAME.lower().replace(' ', '_')}.log"
        else:
            log_file = Path(log_file)
        
        # Main log file (detailed)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)
        
        # Error log file (errors only)
        error_log_file = log_file.parent / f"{log_file.stem}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_log_file,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        error_handler.addFilter(context_filter)
        root_logger.addHandler(error_handler)
    
    # Log initial message
    root_logger.info(f"Logging configured: level={level}, console={console}, file={file_logging}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Module-specific logger
    """
    return logging.getLogger(name)


def set_log_level(level: str, logger_name: Optional[str] = None):
    """
    Set log level for logger.
    
    Args:
        level: Log level string
        logger_name: Logger name (None for root logger)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    logger.setLevel(log_level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(log_level)


def disable_third_party_loggers():
    """
    Reduce verbosity of third-party library loggers.
    """
    # Reduce noise from common libraries
    noisy_loggers = [
        'transformers',
        'torch',
        'PIL',
        'urllib3',
        'werkzeug',
        'uvicorn.access',
        'httpx',
        'asyncio'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


class LoggerContext:
    """
    Context manager for temporary log level changes.
    """
    
    def __init__(self, logger_name: str, level: str):
        """
        Initialize context.
        
        Args:
            logger_name: Logger to modify
            level: Temporary log level
        """
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = getattr(logging, level.upper(), logging.INFO)
    
    def __enter__(self):
        """Enter context."""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.logger.setLevel(self.original_level)


def log_performance(func):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_performance
        def my_function():
            pass
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {elapsed:.3f}s")
            return result
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Failed {func.__name__} after {elapsed:.3f}s: {e}",
                exc_info=True
            )
            raise
    
    return wrapper


def setup_json_logging(enabled: bool = False):
    """
    Setup JSON-structured logging (for production).
    
    Args:
        enabled: Enable JSON logging
    """
    if not enabled:
        return
    
    try:
        import json_log_formatter
        
        # JSON formatter
        json_formatter = json_log_formatter.JSONFormatter()
        
        # Apply to all handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setFormatter(json_formatter)
    
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("json-log-formatter not installed, skipping JSON logging")


# ==================== INITIALIZATION ====================

# Default logger setup (called on import)
def _initialize_default_logging():
    """Initialize default logging configuration."""
    # Only initialize if root logger has no handlers
    if not logging.getLogger().handlers:
        setup_logging()
        disable_third_party_loggers()


# Auto-initialize
_initialize_default_logging()
