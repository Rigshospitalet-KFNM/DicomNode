from dataclasses import dataclass
from logging import DEBUG, Formatter, Logger, LogRecord, NullHandler, StreamHandler
from logging.handlers import TimedRotatingFileHandler, QueueHandler
from pathlib import Path
from multiprocessing.queues import Queue
from sys import stdout
from threading import get_native_id
from traceback import format_exc
from typing import Optional, TextIO

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
  pass

def set_logger(logger: Logger, config: LoggerConfig):
  logger.handlers.clear()

  logger.setLevel(config.log_level)

  if isinstance(config.log_output, TextIO):
    handler = StreamHandler(config.log_output)
  elif isinstance(config.log_output, Path):
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

def queue_logger_thread_target(queue: Queue[LogRecord | None], logger: Logger):
  while True:
    record = queue.get()

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