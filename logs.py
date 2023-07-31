import logging
import logging.handlers

def setup_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # Replace "app.log" with the desired log file name
    logger = setup_logger("app.log")
