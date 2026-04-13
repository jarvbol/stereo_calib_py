import os
import yaml
import logging
from pathlib import Path

def load_config(config_path: str):
    """
    Load configuration from a YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_logging(config):
    """
    Setup logging based on configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing logging settings.
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    import sys
    from logging.handlers import RotatingFileHandler
    
    log_config = config.get('logging', {})
    log_file = log_config.get('file', 'log_intrinsics_camera.log')
    max_bytes = log_config.get('max_bytes', 100000)
    backup_count = log_config.get('backup_count', 10)
    log_level = log_config.get('level', 'DEBUG')
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(log_level, logging.DEBUG)
    
    logging.basicConfig(
        handlers=[
            RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count),
            logging.StreamHandler(sys.stdout)
        ],
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S'
    )
    
    return logging.getLogger(__name__)

def get_calibration_flags(flag_names):
    """
    Convert flag names to actual OpenCV flag values.
    
    Parameters
    ----------
    flag_names : list
        List of flag names as strings.
    
    Returns
    -------
    int
        Combined flags value.
    """
    import cv2 as cv
    
    flags = 0
    for flag_name in flag_names:
        if flag_name.startswith("cv."):
            flag_value = eval(flag_name)
            flags |= flag_value
    
    return flags 