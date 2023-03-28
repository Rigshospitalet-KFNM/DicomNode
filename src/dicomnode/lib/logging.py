__author__ = "Christoffer Vilstrup Jensen"

from logging import Logger
from typing import Optional
from traceback import format_exc


def log_traceback(logger: Logger, exception: Exception, header_message: Optional[str] = None):
  if header_message is not None:
    logger.critical(header_message)

  exception_name_message = f"Encountered exception: {exception.__class__.__name__}"
  logger.critical(exception_name_message)

  traceback_message = format_exc()
  logger.critical(traceback_message)