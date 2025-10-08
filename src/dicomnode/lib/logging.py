""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard library
from dataclasses import dataclass
from logging import Logger, basicConfig, StreamHandler, getLogger, Formatter,\
  DEBUG, INFO, CRITICAL, LogRecord, NullHandler
from logging.handlers import TimedRotatingFileHandler, QueueHandler
from pathlib import Path
from threading import get_native_id, Thread
from multiprocessing import Process, current_process
from multiprocessing.queues import Queue
from io import TextIOWrapper
from sys import stdout, stderr
from os import getpid
from typing import Optional, Union, TextIO
from traceback import format_exc

# Third party packages

# Dicomnode packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER

@dataclass
class LoggingConfiguration:
  queue: Queue


# Global Variables for configuration
__propagate = False
__number_of_backups = 8
__when = "W0"
__date_format = "%Y/%m/%d %H:%M:%S"
__format = "%(asctime)s - %(name)s %(funcName)s %(lineno)s - %(levelname)s - %(message)s"
__log_level = INFO
__logger_name = DICOMNODE_LOGGER_NAME
__log_output: Optional[Union[Path, TextIO]] = stdout
__logger: Optional[Logger] = None
__queue = None

basicConfig(
  format=__format,
  datefmt=__date_format,
  handlers=[StreamHandler(stdout)]
)

def get_logger() -> Logger:
  return getLogger(DICOMNODE_LOGGER_NAME)

def get_response_logger() -> Logger:
  return getLogger(DICOMNODE_PROCESS_LOGGER)

def set_queue_handler(logger: Logger):
  logger.handlers.clear()
  global __log_level
  logger.setLevel(__log_level)

  if __queue is not None:
    logger.addHandler(QueueHandler(__queue))

def set_writer_handler(logger: Logger):
  logger.handlers.clear()
  log_formatter = Formatter(fmt=__format, datefmt=__date_format)

  if isinstance(__log_output, TextIO) or isinstance(__log_output, TextIOWrapper):
    handler = StreamHandler(__log_output)
  elif isinstance(__log_output, Path):
    handler = TimedRotatingFileHandler(
      __log_output,
      when=__when,
      backupCount=__number_of_backups
    )
  elif __log_output is None:
    handler = NullHandler()
  else:
    handler = StreamHandler(stdout)

  handler.setFormatter(log_formatter)
  global __log_level
  logger.addHandler(handler)
  logger.setLevel(__log_level)


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
  global __queue

  dicomnode_logger = getLogger(DICOMNODE_LOGGER_NAME)
  dicomnode_logger.addFilter(__thread_id_filter)
  process_logger = getLogger(DICOMNODE_PROCESS_LOGGER)
  process_logger.addFilter(__thread_id_filter)

  return dicomnode_logger

def __thread_id_filter(record: LogRecord):
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
    queue: Optional[Queue] = None
  ) -> Logger:
  global __log_output
  global __date_format
  global __format
  global __log_level
  global __logger_name
  global __when
  global __number_of_backups
  global __propagate
  global __queue

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

  if queue is not None:
    __queue = queue

  return __setup_logger()

def log_traceback(logger: Logger, exception: Exception, header_message: Optional[str] = None):
  if header_message is not None:
    logger.critical(header_message)

  exception_name_message = f"Encountered exception: {exception.__class__.__name__}"
  logger.critical(exception_name_message)
  traceback_message = format_exc()
  logger.critical(traceback_message)


def listener_logger(queue: Queue[Optional[LogRecord]], logger=None):
  # Do the setup!
  if logger is None:
    logger = getLogger(DICOMNODE_PROCESS_LOGGER)

  set_writer_handler(logger)
  while True:
    record = queue.get()

    if record is None:
      break
    record.name = logger.name
    logger.handle(record)

  queue.close()

def close_thread_logger():
  if __queue is not None:
    __queue.put_nowait(None)
    __queue.join_thread()
