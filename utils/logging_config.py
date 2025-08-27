# utils/logging_config.py
"""
Logging configuration with granular verbosity control.
Usage:
    from utils.logging_config import logger, set_debug_level
    
    set_debug_level('TRACE')  # Shows everything
    set_debug_level('COMPONENT')  # Shows component lifecycle only
    set_debug_level('ERROR')  # Errors only
"""

import logging
import os
from typing import Literal

# Custom logging levels
TRACE_LEVEL = 5        # Extremely verbose (method entry/exit, variable dumps)
COMPONENT_LEVEL = 15   # Component lifecycle events
DEBUG_LEVEL = 20       # Standard debug info
INFO_LEVEL = 20        # Info level (standard)

# Add custom levels to logging
logging.addLevelName(TRACE_LEVEL, "TRACE")
logging.addLevelName(COMPONENT_LEVEL, "COMPONENT")

class CustomLogger(logging.Logger):
    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(TRACE_LEVEL):
            self._log(TRACE_LEVEL, message, args, **kwargs)
    
    def component(self, message, *args, **kwargs):
        if self.isEnabledFor(COMPONENT_LEVEL):
            self._log(COMPONENT_LEVEL, message, args, **kwargs)

# Set the custom logger class
logging.setLoggerClass(CustomLogger)

# Create logger instance
logger = logging.getLogger('wuxia_workbench')

# Debug level mappings
DEBUG_LEVELS = {
    'ERROR': logging.ERROR,      # Errors only
    'WARNING': logging.WARNING,  # Warnings and errors
    'INFO': logging.INFO,        # Info, warnings, errors
    'COMPONENT': COMPONENT_LEVEL, # Component lifecycle + above
    'DEBUG': logging.DEBUG,      # Standard debug + above  
    'TRACE': TRACE_LEVEL        # Everything (very verbose)
}

def set_debug_level(level: Literal['ERROR', 'WARNING', 'INFO', 'COMPONENT', 'DEBUG', 'TRACE'] = 'INFO'):
    """Set the global debug verbosity level."""
    numeric_level = DEBUG_LEVELS.get(level, logging.INFO)
    logger.setLevel(numeric_level)
    
    # Also update all handlers
    for handler in logger.handlers:
        handler.setLevel(numeric_level)
    
    logger.info(f"[LOGGING] Debug level set to: {level} ({numeric_level})")

# Set up handlers for the logger
import sys
from logging.handlers import RotatingFileHandler

if not logger.handlers:
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Rotating file handler
    file_handler = RotatingFileHandler(
        'logs/app.log', 
        mode='a',
        maxBytes=128 * 1024,  # 128KB
        backupCount=5,
        encoding='utf-8',
        delay=True
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Initialize from environment variable or default to INFO
default_level = os.getenv('WUXIA_DEBUG_LEVEL', 'INFO')
set_debug_level(default_level)