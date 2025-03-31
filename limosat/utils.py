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
import pytz

def setup_logging(log_dir='logs', logger_name='LiMOSAT', level=logging.INFO):
    """
    Set up a logger with both console and file handlers.
    
    Parameters:
        log_dir (str): Directory to store log files.
        logger_name (str): Name for the logger.
        level (int): Logging level.
    
    Returns:
        logging.Logger: Configured logger.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{logger_name}_{timestamp}.txt")
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger

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
            logger.info(f"Time taken for {func.__name__}: {exec_time:.3f}s (Image {image_id})")
        else:
            logger.info(f"Time taken for {func.__name__}: {exec_time:.3f}s")
        return result
    return wrapper

def deserialize_descriptors(json_string):
    """
    Deserialize a JSON string back into a numpy array of descriptors.
    """
    return np.array(json.loads(json_string), dtype=np.uint8)

def extract_date(filename):
    """
    Extract the date from the filename and add UTC timezone.
    
    Parameters:
        filename (str): The filename to extract the date from.
    
    Returns:
        datetime or None: The extracted date with UTC timezone, or None if extraction fails.
    """
    base_filename = os.path.basename(filename)
    pattern = r"(\d{8}T\d{6})"
    match = re.search(pattern, base_filename)
    if match:
        date_str = match.group(1)
        try:
            date = datetime.strptime(date_str, "%Y%m%dT%H%M%S")
            date = date.replace(tzinfo=pytz.UTC)
            return date
        except ValueError as ve:
            logger.error(f"Error parsing date from filename '{filename}': {ve}")
            return None
    else:
        return None