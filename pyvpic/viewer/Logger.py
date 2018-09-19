import time
import logging
from functools import wraps
from PyQt5 import QtWidgets

class QLogLabel(QtWidgets.QLabel, logging.Handler):
    """QLogLabel is a simple class to display log messages in a QLabel."""
    def emit(self, record):
        """Display the most recent log message."""
        self.setText(self.format(record).strip())

def log_timing(func):
    """Decorator to add logging to a class method. arg[0] should be a class
    instance and args[0].logger should exist. If not, this logs to root."""
    @wraps(func)
    def with_logging(*args, **kwargs):
        """Inner function with logging."""
        if not args or not hasattr(args[0], 'logger'):
            logger = logging.getLogger('')
        else:
            logger = args[0].logger

        logger.info(f'Starting {func.__name__} ...')
        start_time = time.time()
        retval = func(*args, **kwargs)
        stop_time = time.time()
        logger.info(f'\t{func.__name__} completed in '
                    f'{stop_time-start_time:0.3f} s.')
        return retval
    return with_logging
