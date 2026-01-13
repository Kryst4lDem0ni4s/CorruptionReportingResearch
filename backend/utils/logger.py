"""
Logger Utils - Logging utilities and helpers

Provides:
- Structured logging setup
- Logger configuration
- Context-based logging
- Performance logging
- Log formatting
"""

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# Try to import colorama for colored console output (optional)
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Fallback: empty strings
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = DIM = RESET_ALL = ''


# Global logger cache
_logger_cache: Dict[str, logging.Logger] = {}


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for better readability.
    
    Uses colors to distinguish log levels:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Bright Red
    """
    
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        """
        Initialize colored formatter.
        
        Args:
            fmt: Log format string
            datefmt: Date format string
        """
        if fmt is None:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if datefmt is None:
            datefmt = '%Y-%m-%d %H:%M:%S'
        
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.
        
        Args:
            record: Log record
            
        Returns:
            Formatted log string
        """
        if not HAS_COLORAMA:
            return super().format(record)
        
        # Get color for level
        color = self.COLORS.get(record.levelno, '')
        
        # Format the message
        formatted = super().format(record)
        
        # Add color
        if color:
            formatted = f"{color}{formatted}{Style.RESET_ALL}"
        
        return formatted


class StructuredFormatter(logging.Formatter):
    """
    JSON-structured formatter for machine-readable logs.
    
    Outputs logs in JSON format with structured fields.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            JSON formatted log string
        """
        import json
        
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        return json.dumps(log_data)


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[Path] = None,
    colored: bool = True,
    structured: bool = False
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        colored: Use colored console output
        structured: Use JSON structured logging
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if structured:
        console_formatter = StructuredFormatter()
    elif colored and HAS_COLORAMA:
        console_formatter = ColoredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        # Always use structured format for file logs
        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Log the setup
    root_logger.info(f"Logging initialized at {level} level")
    if log_file:
        root_logger.info(f"Logging to file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the specified name.
    
    Uses caching to avoid creating duplicate loggers.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    if name not in _logger_cache:
        logger = logging.getLogger(name)
        _logger_cache[name] = logger
    
    return _logger_cache[name]


class LoggerContext:
    """
    Context manager for logging with additional context.
    
    Adds extra fields to log records within the context.
    
    Example:
        with LoggerContext(logger, submission_id='sub123'):
            logger.info("Processing submission")
            # Logs: "Processing submission [submission_id=sub123]"
    """
    
    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize logger context.
        
        Args:
            logger: Logger instance
            **context: Context key-value pairs
        """
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Enter context."""
        # Save old factory
        self.old_factory = logging.getLogRecordFactory()
        
        # Create new factory with context
        context = self.context
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            record.extra_data = context
            
            # Add context to message
            if context:
                context_str = ', '.join(f'{k}={v}' for k, v in context.items())
                record.msg = f"{record.msg} [{context_str}]"
            
            return record
        
        logging.setLogRecordFactory(record_factory)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # Restore old factory
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


@contextmanager
def log_performance(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO
):
    """
    Context manager for performance logging.
    
    Logs the duration of an operation.
    
    Args:
        logger: Logger instance
        operation: Operation description
        level: Log level
        
    Example:
        with log_performance(logger, "Database query"):
            result = query_database()
        # Logs: "Database query completed in 1.23s"
    """
    start_time = time.time()
    
    logger.log(level, f"{operation} started")
    
    try:
        yield
        
        duration = time.time() - start_time
        logger.log(level, f"{operation} completed in {duration:.2f}s")
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"{operation} failed after {duration:.2f}s: {e}",
            exc_info=True
        )
        raise


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Optional[Exception] = None
):
    """
    Log exception with traceback.
    
    Args:
        logger: Logger instance
        message: Error message
        exc: Exception instance (optional)
    """
    if exc:
        logger.error(f"{message}: {exc}", exc_info=True)
    else:
        logger.error(message, exc_info=True)


def log_dict(
    logger: logging.Logger,
    data: Dict[str, Any],
    level: int = logging.DEBUG,
    prefix: str = ''
):
    """
    Log dictionary in readable format.
    
    Args:
        logger: Logger instance
        data: Dictionary to log
        level: Log level
        prefix: Optional prefix for each line
    """
    if prefix:
        logger.log(level, prefix)
    
    for key, value in data.items():
        logger.log(level, f"  {key}: {value}")


def set_log_level(logger: logging.Logger, level: str):
    """
    Set logger level.
    
    Args:
        logger: Logger instance
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)


def disable_logger(name: str):
    """
    Disable specific logger.
    
    Args:
        name: Logger name
    """
    logger = logging.getLogger(name)
    logger.disabled = True


def enable_logger(name: str):
    """
    Enable specific logger.
    
    Args:
        name: Logger name
    """
    logger = logging.getLogger(name)
    logger.disabled = False


# Convenience functions for quick logging

def debug(message: str, logger_name: str = 'root'):
    """Quick debug log."""
    logging.getLogger(logger_name).debug(message)


def info(message: str, logger_name: str = 'root'):
    """Quick info log."""
    logging.getLogger(logger_name).info(message)


def warning(message: str, logger_name: str = 'root'):
    """Quick warning log."""
    logging.getLogger(logger_name).warning(message)


def error(message: str, logger_name: str = 'root'):
    """Quick error log."""
    logging.getLogger(logger_name).error(message)


def critical(message: str, logger_name: str = 'root'):
    """Quick critical log."""
    logging.getLogger(logger_name).critical(message)


# Module-level logger for this file
logger = get_logger(__name__)
