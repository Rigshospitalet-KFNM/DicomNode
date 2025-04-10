""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard library
from logging import Logger, basicConfig, StreamHandler, getLogger, Formatter,\
  DEBUG, INFO, CRITICAL, LogRecord
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from threading import get_native_id
from multiprocessing import Process
from multiprocessing.queues import Queue
from io import TextIOWrapper
from sys import stdout
from typing import Optional, Union, TextIO
from traceback import format_exc

# Third party packages

# Dicomnode packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME



# Global Variables for configuration
__propagate = False
__number_of_backups = 8
__when = "W0"
__date_format = "%Y/%m/%d %H:%M:%S"
__format = "%(asctime)s - %(name)s %(funcName)s %(lineno)s - %(levelname)s - %(message)s"
__log_level = INFO
__logger_name = DICOMNODE_LOGGER_NAME
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
  __logger.addFilter(__thread_id_filter)

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
  else:
    handler = StreamHandler(stdout)

  log_formatter = Formatter(fmt=__format, datefmt=__date_format)
  handler.setFormatter(log_formatter)
  if __logger.hasHandlers():
    __logger.handlers.clear()
  __logger.propagate = __propagate
  __logger.addHandler(handler)

  return __logger

def __thread_id_filter(record):
  record.thread_id = get_native_id()
  return record

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

def listener_configuration():
  return __setup_logger()

def listener_logger(queue: Queue[LogRecord]):
  while True:
    try:
      record = queue.get()
      if record is None:
        break
      logger = getLogger(record.name)
      logger.handle(record)

    except Exception:
      import sys, traceback
      print("Wopsy", file=sys.stderr)
      traceback.print_exc(file=sys.stderr)

def _setup_logger2(queue: Queue[LogRecord]):
  listener_process = Process(target=listener_logger, args=(queue,), daemon=True)
  listener_process.start()
