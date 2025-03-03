import logging

# Global logger instance
_logger = None


def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Configure logging for the entire application
    """
    global _logger

    # Create logger
    _logger = logging.getLogger("simulation")
    _logger.setLevel(log_level)

    # Clear any existing handlers
    if _logger.hasHandlers():
        _logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    return _logger


def get_logger():
    """
    Get the configured logger instance
    """
    global _logger
    if _logger is None:
        # Configure with defaults if not already configured
        _logger = logging.getLogger("simulation")
        _logger.setLevel(logging.INFO)

        # Add a default console handler if no handlers exist
        if not _logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            _logger.addHandler(handler)

    return _logger
