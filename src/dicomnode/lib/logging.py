__author__ = "Christoffer Vilstrup Jensen"

from logging import Logger, basicConfig, StreamHandler, getLogger, Formatter,\
  DEBUG, INFO, WARNING, ERROR, CRITICAL, NullHandler
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from io import TextIOBase, TextIOWrapper
from sys import stdout
from typing import Optional, Union, TextIO
from traceback import format_exc

# Global Variables for configuration
__propagate = False
__number_of_backups = 8
__when = "W0"
__date_format = "%Y/%m/%d %H:%M:%S"
__format = "%(asctime)s - %(name)s %(funcName)s %(lineno)s - %(levelname)s - %(message)s"
__log_level = INFO
__logger_name = "dicomnode"
__log_output: Optional[Union[Path, TextIO]] = stdout
__logger = None

basicConfig(
  format=__format,
  datefmt=__date_format,
  handlers=[StreamHandler(stdout)]
)


def get_logger() -> Logger:
  global __logger
  if __logger is not None:
    return __logger

  return __setup_logger()

def __setup_logger() -> Logger:
  global __log_level
  global __logger
  global __logger_name
  global __date_format
  global __format
  global __log_output
  global __when
  global __number_of_backups
  global __propagate

  __logger = getLogger(__logger_name)
  __logger.setLevel(__log_level)


  if __log_output is None:
    __logger.setLevel(CRITICAL + 1)
    return __logger

  elif isinstance(__log_output, TextIO) or isinstance(__log_output, TextIOWrapper):
    handler = StreamHandler(__log_output)

  elif isinstance(__log_output, Path):
    handler = TimedRotatingFileHandler(
      __log_output,
      when=__when,
      backupCount=__number_of_backups
    )

  log_formatter = Formatter(fmt=__format, datefmt=__date_format)
  handler.setFormatter(log_formatter)
  if __logger.hasHandlers():
    __logger.handlers.clear()
  __logger.propagate = __propagate
  __logger.addHandler(handler)

  return __logger


def set_logger(
    log_output: Optional[Union[TextIO, Path]],
    log_level: Optional[int] = None,
    format: Optional[str] = None,
    date_format: Optional[str] = None,
    logger_name: Optional[str] = None,
    when: Optional[str] = None,
    backupCount: Optional[int] = None,
    propagate: Optional[bool]  = None,
  ) -> Logger:
  global __log_output
  global __date_format
  global __format
  global __log_level
  global __logger_name
  global __when
  global __number_of_backups
  global __propagate

  __log_output = log_output

  if log_level is not None:
    __log_level = log_level

  if format is not None:
    __format = format

  if date_format is not None:
    __date_format = date_format

  if logger_name is not None:
    __logger_name = logger_name

  if when is not None:
    __when = when

  if backupCount is not None:
    __number_of_backups = backupCount

  if propagate is not None:
    __propagate = propagate

  return __setup_logger()

def log_traceback(logger: Logger, exception: Exception, header_message: Optional[str] = None):
  if header_message is not None:
    logger.critical(header_message)

  exception_name_message = f"Encountered exception: {exception.__class__.__name__}"
  logger.critical(exception_name_message)
  traceback_message = format_exc()
  logger.critical(traceback_message)