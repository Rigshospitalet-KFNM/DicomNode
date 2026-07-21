from dataclasses import dataclass
from logging import DEBUG, Formatter, Logger, LogRecord, Handler, NullHandler,\
  StreamHandler, getLogger
from logging.handlers import TimedRotatingFileHandler, QueueHandler
from pathlib import Path
import multiprocessing
multiprocessing_context = multiprocessing.get_context('spawn')
import os
from io import TextIOBase
from multiprocessing.queues import Queue
from queue import Empty
from sys import stdout
from threading import get_native_id, Thread
from traceback import format_exc
from typing import Optional, TextIO

from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER
from dicomnode.config import DicomnodeConfig
from dicomnode.lib.exceptions import ContractViolation

@dataclass
class LoggerConfig:
  log_level: int = DEBUG
  date_format:str  = "%Y/%m/%d %H:%M:%S"
  format: str = "%(asctime)s - %(name)s %(funcName)s %(lineno)s - %(levelname)s - %(message)s"
  log_output: Path | TextIO | None | Queue = stdout
  when: str = "W0"
  number_of_backups: int = 8
  propagate: bool = False

def get_logger():
  return getLogger(DICOMNODE_LOGGER_NAME)

def set_logger(logger: Logger, config: LoggerConfig, handler_: Optional[Handler] = None ):
  logger.handlers.clear()

  logger.setLevel(config.log_level)

  if handler_ is not None:
    handler = handler_
  elif isinstance(config.log_output, TextIOBase):
    handler = StreamHandler(config.log_output)
  elif isinstance(config.log_output, Path) or isinstance(config.log_output, str):
    handler = TimedRotatingFileHandler(
      config.log_output,
      when=config.when,
      backupCount=config.number_of_backups
    )
  elif isinstance(config.log_output, Queue):
    handler = QueueHandler(config.log_output)
  else: # config is None
    handler = NullHandler()

  formatter = Formatter(
    fmt=config.format,
    datefmt=config.date_format
  )

  handler.setFormatter(formatter)

  logger.addHandler(handler)
  logger.addFilter(_thread_id_filter)

def queue_logger_thread_target(queue: Queue[LogRecord | None]):
  logger = get_logger()

  while True:
    try:
      record = queue.get(timeout=0.1)
    except Empty:
      continue

    if record is None:
      break

    record.name = logger.name
    logger.handle(record)

  queue.close()


def log_traceback(logger: Logger, exception: Exception, header_message: Optional[str] = None):
  if header_message is not None:
    logger.critical(header_message)

  exception_name_message = f"Encountered exception: {exception.__class__.__name__}"
  logger.critical(exception_name_message)
  traceback_message = format_exc()
  logger.critical(traceback_message)


def _thread_id_filter(record: LogRecord):
  record.thread_id = get_native_id()
  return record

class LogManager:
  def __init__(self, config: DicomnodeConfig, existing_queue: Optional[Queue[LogRecord | None]] ) -> None:
    """The Log manager is the class that is responsible for managing the
    handlers of the various loggers in dicomnode.


    Args:
        config (DicomnodeConfig): _description_
        existing_queue (Optional[Queue[LogRecord  |  None]]): _description_
    """
    self.dicomnode_config = config
    self.logging_config = self.get_logging_config()
    self._log_queue = existing_queue
    self._owning_queue = existing_queue is None
    self._logging_thread: Optional[Thread] = None
    self.handler = self.destination_handler()

    set_logger(self.get_logger(), self.get_logging_config(), self.handler)

  def get_logger(self) -> Logger:
    return getLogger(DICOMNODE_LOGGER_NAME)

  def get_process_logger(self) -> Logger:
    return getLogger(DICOMNODE_PROCESS_LOGGER) if self.should_queue_log() else self.get_logger()

  def start_queue(self):
    if not self.should_queue_log():
      return

    if self._logging_thread is not None:
      raise ContractViolation("Cannot start queue logging when you're already queue logging!")

    if self._log_queue is None:
      self._log_queue = multiprocessing_context.Queue()

    self._logging_thread = Thread(
      target=queue_logger_thread_target,
      args=(self._log_queue,),
      name="Log Queue Reader Thread",
    )

    self._logging_thread.start()

  def stop_queue(self):
    if self._log_queue is None or self._logging_thread is None or not self.should_queue_log():
      return

    self.get_logger().info("Closing Log queue!")

    self._log_queue.put_nowait(None)
    self._logging_thread.join()
    self._log_queue.join_thread()

    self._log_queue = None
    self._logging_thread = None

  def get_logging_config(self):
    if isinstance(self.dicomnode_config.LOG_OUTPUT, str):
      if self.dicomnode_config.LOG_OUTPUT == "stdout":
        output = stdout
      else:
        output = Path(self.dicomnode_config.LOG_OUTPUT)
    elif isinstance(self.dicomnode_config.LOG_OUTPUT, TextIOBase):
      output = self.dicomnode_config.LOG_OUTPUT
    else:
      output = None

    return LoggerConfig(
      log_level=self.dicomnode_config.LOG_LEVEL,
      date_format=self.dicomnode_config.LOG_DATE_FORMAT,
      format=self.dicomnode_config.LOG_FORMAT,
      log_output=output,
      when=self.dicomnode_config.LOG_WHEN,
      number_of_backups=self.dicomnode_config.LOG_NUMBER_OF_BACK_UPS,
    )

  def queue_logging_config(self):
    return LoggerConfig(
      log_level=self.dicomnode_config.LOG_LEVEL,
      date_format=self.dicomnode_config.LOG_DATE_FORMAT,
      format=self.dicomnode_config.LOG_FORMAT,
      log_output=self._log_queue,
      when=self.dicomnode_config.LOG_WHEN,
      number_of_backups=self.dicomnode_config.LOG_NUMBER_OF_BACK_UPS
    )

  def should_queue_log(self) -> bool:
    return bool(self.dicomnode_config.PROCESSING_DIRECTORY)

  def destination_handler(self):
    if isinstance(self.logging_config.log_output, TextIOBase):
      return StreamHandler(self.logging_config.log_output)
    elif isinstance(self.logging_config.log_output, Path) or isinstance(self.logging_config.log_output, str):
      return TimedRotatingFileHandler(
        self.logging_config.log_output,
        when=self.logging_config.when,
        backupCount=self.logging_config.number_of_backups
      )
    else: # config is None
      return NullHandler()

  def get_queue_handler(self):
    if isinstance(self._log_queue, Queue):
      return QueueHandler(self._log_queue)
    return NullHandler()