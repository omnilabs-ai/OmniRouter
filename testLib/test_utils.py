import logging
import os
import traceback
import sys
from datetime import datetime
from functools import wraps

def setup_logger(log_level=logging.INFO):
    """
    Setup a logger with configurable log level and proper Unicode handling
    Args:
        log_level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'test_run_{timestamp}.log')

    # Configure logger
    logger = logging.getLogger('test_logger')
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers = []

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # Console handler with proper encoding for Windows
    if sys.platform == 'win32':
        # Force UTF-8 encoding for Windows console
        sys.stdout.reconfigure(encoding='utf-8')
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        console_handler = logging.StreamHandler()
        
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Example usage of different log levels:
# DEBUG: Detailed information for debugging
# INFO: General information about program execution
# WARNING: Indicates a potential problem
# ERROR: A more serious problem
# CRITICAL: A critical error that may prevent the program from running

# Create a global logger instance with default INFO level
test_logger = setup_logger(logging.DEBUG)

# To use a different level, call setup_logger with the desired level:
# test_logger = setup_logger(logging.DEBUG)  # For debug level
# test_logger = setup_logger(logging.WARNING)  # For warning level 