"""
Logging system for the Research Assistant RAG application.

For beginners: Logging is like keeping a diary of what your application does.
It helps you understand what's happening and debug problems.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional
from pathlib import Path
import json

from core.config import settings


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    For beginners: This formats log messages as JSON, making them easier
    to search and analyze in production systems.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Convert log record to JSON format."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = getattr(record, 'user_id', None)
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = getattr(record, 'request_id', None)
        if hasattr(record, 'query'):
            log_entry["query"] = getattr(record, 'query', None)
        if hasattr(record, 'duration'):
            log_entry["duration"] = getattr(record, 'duration', None)
            
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    
    For beginners: This adds colors to console logs, making them easier to read.
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Add colors to log messages."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging() -> None:
    """
    Set up the logging configuration for the application.
    
    For beginners: This function configures how and where logs are written.
    We'll log to both console (for development) and files (for production).
    """
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # ==============================================
    # Console Handler (for development)
    # ==============================================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    if settings.development:
        # Use colored formatter for development
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        # Use simple formatter for production
        console_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # ==============================================
    # File Handler (for persistent logging)
    # ==============================================
    file_handler = logging.FileHandler(settings.log_file)
    file_handler.setLevel(logging.DEBUG)
    
    if settings.log_format.lower() == 'json':
        # Use JSON formatter for structured logging
        file_formatter = JSONFormatter()
    else:
        # Use standard formatter
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s'
        )
    
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # ==============================================
    # Configure specific loggers
    # ==============================================
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # Set our application loggers
    logging.getLogger("research_assistant").setLevel(logging.DEBUG)
    logging.getLogger("core").setLevel(logging.DEBUG)
    logging.getLogger("services").setLevel(logging.DEBUG)
    logging.getLogger("utils").setLevel(logging.DEBUG)
    
    # Log the setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized - Level: {settings.log_level}, Format: {settings.log_format}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    For beginners: This function creates a logger for a specific module.
    Each part of your application can have its own logger.
    
    Args:
        name: The name of the logger (usually __name__)
        
    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, duration: float, **kwargs) -> None:
    """
    Log a function call with timing information.
    
    For beginners: This helps track performance of important functions.
    
    Args:
        func_name: Name of the function being called
        duration: How long the function took to execute
        **kwargs: Additional context information
    """
    logger = get_logger("research_assistant.performance")
    
    # Create log record with extra fields
    extra_info = {
        "function": func_name,
        "duration": duration,
        **kwargs
    }
    
    logger.info(f"Function '{func_name}' completed in {duration:.3f}s", extra=extra_info)


def log_query(query: str, user_id: Optional[str] = None, request_id: Optional[str] = None) -> None:
    """
    Log a user query for analytics.
    
    For beginners: This helps track what users are asking about.
    
    Args:
        query: The user's search query
        user_id: Optional user identifier
        request_id: Optional request identifier
    """
    logger = get_logger("research_assistant.queries")
    
    extra_info = {
        "query": query,
        "user_id": user_id,
        "request_id": request_id
    }
    
    logger.info(f"User query: '{query}'", extra=extra_info)


def log_error(error: Exception, context: Optional[dict] = None) -> None:
    """
    Log an error with additional context.
    
    For beginners: This helps track errors that occur in the system.
    
    Args:
        error: The exception that occurred
        context: Additional context information
    """
    logger = get_logger("research_assistant.errors")
    
    context = context or {}
    
    logger.error(
        f"Error occurred: {type(error).__name__}: {str(error)}", 
        exc_info=error,
        extra=context
    )


def log_performance_metrics(metrics: dict) -> None:
    """
    Log performance metrics.
    
    For beginners: This tracks how well the system is performing.
    
    Args:
        metrics: Dictionary of performance metrics
    """
    logger = get_logger("research_assistant.metrics")
    
    logger.info("Performance metrics", extra=metrics)


# ==============================================
# Context Manager for Function Timing
# ==============================================

class FunctionTimer:
    """
    Context manager for timing function execution.
    
    For beginners: This is a tool to automatically measure how long
    functions take to execute.
    
    Usage:
        with FunctionTimer("my_function"):
            # Your code here
            pass
    """
    
    def __init__(self, function_name: str, **context):
        self.function_name = function_name
        self.context = context
        self.start_time = None
        self.logger = get_logger("research_assistant.timing")
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.function_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type is None:
                # Function completed successfully
                log_function_call(self.function_name, duration, **self.context)
            else:
                # Function failed
                self.logger.error(
                    f"Function '{self.function_name}' failed after {duration:.3f}s",
                    exc_info=(exc_type, exc_val, exc_tb),
                    extra={"function": self.function_name, "duration": duration, **self.context}
                )


# ==============================================
# Example Usage (for testing)
# ==============================================

if __name__ == "__main__":
    # Test the logging system
    print("Testing logging system...")
    
    setup_logging()
    
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test function timing
    with FunctionTimer("test_function", user_id="test_user"):
        import time
        time.sleep(0.1)  # Simulate work
    
    # Test query logging
    log_query("What is machine learning?", user_id="test_user", request_id="req_123")
    
    # Test error logging
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        log_error(e, {"context": "testing error logging"})
    
    print("Logging test completed!") 