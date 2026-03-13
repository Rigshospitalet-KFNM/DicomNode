"""This modules is for creating new threads and processes inside of the
library"""

# Python standard library
from datetime import datetime
from enum import Enum
from logging import getLogger
from threading import Thread
import multiprocessing

# Dicomnode modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME



MULTIPROCESSING = multiprocessing.get_context('spawn')
Process = MULTIPROCESSING.Process

class ParallelPrimitive(Enum):
  THREAD = 1
  PROCESS = 2

class ProcessLikeThread(Thread):
  def __init__(self, group=None, target=None, name=None,
              args=(), kwargs={}, daemon=True):
    super().__init__(group, target, name, args, kwargs, daemon=daemon)
    self._return = None
    self.exception = None

  def run(self):
    try:
      self._return = self._target(*self._args, **self._kwargs) #type: ignore
    except Exception as e:
      self.exception = e


  def join(self, timeout: float | None=None): #type: ignore
    super().join(timeout)
    return self._return

def spawn_thread(thread_function, *args, name=None, **kwargs):
  logger = kwargs['logger'] if 'logger' in kwargs else getLogger("dicomnode")
  thread = ProcessLikeThread(
    target=thread_function, args=args, name=name, kwargs=kwargs
  )

  thread.start()

  log_message = f"Spawned Thread {thread.native_id} with {thread_function.__name__} - {name}"

  logger.debug(log_message)

  return thread

def spawn_process(process_function, *args, start=True,name=None, context=None, **kwargs):
  logger = kwargs['logger'] if 'logger' in kwargs else getLogger("dicomnode")

  if context is None:
    context = multiprocessing.get_context('spawn')

  process = context.Process(
    target=process_function, args=args, name=name
  )

  if start:
    process.start()

  log_message = f"Spawned Process {process.pid} with {process_function.__name__}"

  logger.debug(log_message)

  return process


class Parallel:
  primitive: ProcessLikeThread | Process
  def __init__(self, primitive: ParallelPrimitive, target, *args, **kwargs) -> None:
    if primitive == ParallelPrimitive.PROCESS:
      self.primitive = spawn_process(target, *args, **kwargs, context=MULTIPROCESSING)
    elif primitive == ParallelPrimitive.THREAD:
      self.primitive = spawn_thread(target, *args, **kwargs)
    else: # pragma: no cover
      raise TypeError("primitive argument is not of the type ParallelPrimitive")

  def join(self):
    logger = getLogger(DICOMNODE_LOGGER_NAME)
    timeouts = 0
    timeout_duration_seconds = 2.0
    started_joined = datetime.now()

    while self.primitive.is_alive():
      self.primitive.join(timeout_duration_seconds)
      if self.primitive.is_alive():
        timeouts += 1
        timeout_duration_seconds = (timeout_duration_seconds * 2)
        logger.info(f"{self.primitive_name()} - ({started_joined}) - encountered timeout {timeouts}")

  def id(self):
    if isinstance(self.primitive, Process):
      return self.primitive.pid
    elif isinstance(self.primitive, Thread):
      return self.primitive.native_id
    else: # pragma: no cover
      logger = getLogger(DICOMNODE_LOGGER_NAME)
      logger.critical(f"The Primitive is not a thread or Process")
      return -1

  def is_successful(self):
    if self.primitive.is_alive():
      raise RuntimeError("Parallel Primitive is still alive, join first")

    match self.primitive:
      case Process():
        return self.primitive.exitcode == 0
      case ProcessLikeThread():
        return self.primitive.exception is None

  def primitive_name(self):
    match self.primitive:
      case Process():
        return "Process"
      case ProcessLikeThread():
        return "Thread"
    # pragma: no cover
    logger = getLogger(DICOMNODE_LOGGER_NAME)
    logger.critical(f"The Primitive is not a thread or Process")
    return "Unknown"
