import time
import inspect
import logging

# Track already given warnings.
_GIVEN_WARNINGS = {}

# Configure the logger.
logging.basicConfig(
    format = '[AgML] %(asctime)s %(levelname)s - %(name)s: %(message)s',
    datefmt = '%m-%d-%Y %H:%M:%S')

def log(msg, level = logging.WARNING):
    """Logs a message to the console.

    This is an internal wrapper method for the logging module,
    that takes care of the different logging warning levels
    and ensures that no log is repeated twice at the same time.
    """
    method_map = {
        logging.DEBUG: logging.debug,
        logging.INFO: logging.info,
        logging.WARNING: logging.warning,
        logging.CRITICAL: logging.critical
    }

    # Check if the warning has been delivered recently.
    log_method = method_map[level]
    save_hash = f"{round(time.time(), -1)}:{inspect.stack()[1].function}"
    if save_hash in _GIVEN_WARNINGS.keys():
        if _GIVEN_WARNINGS[save_hash] == msg:
            save_time = float(save_hash.split(':')[0])
            if time.time() - save_time > 240:
                log_method(msg)
    else:
        log_method(msg)
    _GIVEN_WARNINGS[save_hash] = msg






