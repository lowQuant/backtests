import logging
import sys
from datetime import datetime

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Suppress ib_async logs
logging.getLogger("ib_async").setLevel(logging.ERROR)

logger = logging.getLogger("backtests")

def add_log(message: str, level: str = "info") -> None:
    """
    Add a log message.
    """
    if level.lower() == "debug":
        logger.debug(message)
    elif level.lower() == "warning":
        logger.warning(message)
    elif level.lower() == "error":
        logger.error(message)
    else:
        logger.info(message)
