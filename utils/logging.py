import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    """Sets up a logger that writes to both console and a file with rotation."""
    logger = logging.getLogger('wuxia_workbench')
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Rotating file handler - keep logs under 64KB (~15k tokens)
        file_handler = RotatingFileHandler(
            'logs/app.log', 
            mode='a',
            maxBytes=64 * 1024,  # 64KB ≈ 15k GPT tokens
            backupCount=10,      # Keep 10 old logs
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
