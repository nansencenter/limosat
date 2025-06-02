# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

"""
utils.py

Utility functions for limosat
"""

import os
import logging
from datetime import datetime
import functools
import time
import json
import numpy as np
import re
import pandas as pd

APP_LOGGER_NAME = 'LiMOSAT'

def setup_logging(log_dir='logs',
                  logger_name=APP_LOGGER_NAME, # Use the authoritative name
                  console_level=logging.INFO,
                  file_level=logging.DEBUG,
                  filename_prefix=APP_LOGGER_NAME): # Filename prefix defaults to logger name
    """
    Sets up basic logging with console and file handlers.
    Ensures handlers are added only once to the specified logger.
    """
    logger_instance = logging.getLogger(logger_name)

    # Set the overall level for the logger.
    logger_instance.setLevel(min(console_level, file_level))

    # Only add handlers if THIS logger instance has none
    if not logger_instance.handlers:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.log")

        # Console Handler
        console_h = logging.StreamHandler()
        console_h.setLevel(console_level)

        # File Handler
        file_h = logging.FileHandler(log_file_path)
        file_h.setLevel(file_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-8s - %(message)s')
        console_h.setFormatter(formatter)
        file_h.setFormatter(formatter)

        logger_instance.addHandler(console_h)
        logger_instance.addHandler(file_h)
        
        logger_instance.log_file_path = log_file_path

    return logger_instance

# Initialize a module-level logger using our setup
logger = setup_logging()

def log_execution_time(func):
    """
    Decorator to log the execution time of functions and methods.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        exec_time = time.time() - start_time
        
        # Determine if this is an instance method call with a 'current_image_id' attribute.
        if args and hasattr(args[0], '__class__') and func.__name__ in args[0].__class__.__dict__:
            image_id = getattr(args[0], 'current_image_id', None)
        else:
            image_id = None
        
        if image_id is not None:
            logger.debug(f"Time taken for {func.__name__}: {exec_time:.3f}s (Image {image_id})")
        else:
            logger.debug(f"Time taken for {func.__name__}: {exec_time:.3f}s")
        return result
    return wrapper   

def extract_date(filename):
    """
    Extract the date from the filename, convert to pandas timestam
    
    Parameters:
        filename (str): The filename to extract the date from.
    
    Returns:
        datetime or None: The extracted date
    """
    base_filename = os.path.basename(filename)
    pattern = r"(\d{8}T\d{6})"
    match = re.search(pattern, base_filename)
    if match:
        date_str = match.group(1)
        try:
            date = datetime.strptime(date_str, "%Y%m%dT%H%M%S")
            date = pd.Timestamp(date)
            return date
        except ValueError as ve:
            logger.error(f"Error parsing date from filename '{filename}': {ve}")
            return None
    else:
        return None