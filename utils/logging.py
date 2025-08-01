import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# Import our custom logging configuration
try:
    from .logging_config import logger, set_debug_level
    CUSTOM_LOGGING = True
except ImportError:
    CUSTOM_LOGGING = False

def setup_logger():
    """Sets up a logger that writes to both console and a file with rotation."""
    if CUSTOM_LOGGING:
        # Use the enhanced logging system
        return logger
    
    # Fallback to basic logging
    logger = logging.getLogger('wuxia_workbench')
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Rotating file handler - keep logs under 128KB (optimal for Claude analysis)
        file_handler = RotatingFileHandler(
            'logs/app.log', 
            mode='a',
            maxBytes=128 * 1024,  # 128KB ≈ 30k tokens, readable by Claude
            backupCount=5,        # Keep 5 old logs for debugging history
            encoding='utf-8',
            delay=True
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG) # Verbose console output
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger()

# Export the set_debug_level function if available
if CUSTOM_LOGGING:
    __all__ = ['logger', 'set_debug_level']
else:
    __all__ = ['logger']
