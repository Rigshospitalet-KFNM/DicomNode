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



def spawn_thread(thread_function, *args, name=None, **kwargs):
  logger = kwargs['logger'] if 'logger' in kwargs else getLogger("dicomnode")
  thread = Thread(
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
  primitive: Thread | Process
  def __init__(self, primitive: ParallelPrimitive, target, *args, **kwargs) -> None:
    if primitive == ParallelPrimitive.PROCESS:
      self.primitive = spawn_process(target, *args, **kwargs, context=MULTIPROCESSING)
    elif primitive == ParallelPrimitive.THREAD:
      self.primitive = spawn_thread(target, *args, **kwargs)
    else:
      raise ValueError("")

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
    else:
      logger = getLogger(DICOMNODE_LOGGER_NAME)
      logger.critical(f"The Primitive is not a thread or Process")

  def primitive_name(self):
    if isinstance(self.primitive, Process):
      return "Process"
    elif isinstance(self.primitive, Thread):
      return "Thread"
    else:
      logger = getLogger(DICOMNODE_LOGGER_NAME)
      logger.critical(f"The Primitive is not a thread or Process")
      return "Unknown"
