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
import re
import pandas as pd
import yaml

APP_LOGGER_NAME = 'LiMOSAT'

def setup_logging(log_dir='logs',
                  logger_name=APP_LOGGER_NAME,
                  console_level=logging.INFO,
                  file_level=logging.DEBUG,
                  filename_prefix=APP_LOGGER_NAME,
                  persist_log=False):
    """
    Sets up logging.

    If persist_log is True, logs to both console and a file.
    If persist_log is False, logs only to the console at DEBUG level.

    Ensures handlers are added only once to the specified logger, and clears
    existing handlers to support interactive notebook environments.
    """
    logger_instance = logging.getLogger(logger_name)
    
    # Force clearing of existing handlers for interactive notebook use
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    # If not persisting, we want the full log (DEBUG level) in the console.
    effective_console_level = logging.DEBUG if not persist_log else console_level

    # Set the overall level for the logger.
    log_level = min(effective_console_level, file_level) if persist_log else effective_console_level
    logger_instance.setLevel(log_level)

    # Add console handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-8s - %(message)s')
    console_h = logging.StreamHandler()
    console_h.setLevel(effective_console_level)
    console_h.setFormatter(formatter)
    logger_instance.addHandler(console_h)

    # Add file handler only if persisting
    if persist_log:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.log")

        file_h = logging.FileHandler(log_file_path)
        file_h.setLevel(file_level)
        file_h.setFormatter(formatter)
        logger_instance.addHandler(file_h)
        
        logger_instance.log_file_path = log_file_path
    else:
        logger_instance.log_file_path = None

    return logger_instance

# Initialize a module-level logger.
# This logger will be configured by the application using setup_logging.
# By default, it will have no handlers and will not output messages.
logger = logging.getLogger(APP_LOGGER_NAME)
# To prevent messages from being propagated to the root logger if no handlers are configured
logger.propagate = False
# Add a NullHandler to prevent "No handlers could be found" warnings.
logger.addHandler(logging.NullHandler())

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


def load_config(config_path):
    """
    Loads a YAML configuration file.

    Parameters:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise
