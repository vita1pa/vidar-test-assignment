import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger with a specific name and log file."""

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    directory = os.path.dirname(log_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger