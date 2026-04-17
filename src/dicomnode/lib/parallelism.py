"""This modules is for creating new threads and processes inside of the
library"""

# Python standard library
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from io import StringIO
from logging import getLogger
import multiprocessing
import sys
from threading import Thread
from typing import Optional

# Dicomnode modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME

@dataclass
class ProgramOutput:
  stdout : str
  stderr : str



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
  logger = kwargs['logger'] if 'logger' in kwargs else getLogger(DICOMNODE_LOGGER_NAME)
  thread = ProcessLikeThread(
    target=thread_function, args=args, name=name, kwargs=kwargs
  )

  thread.start()

  log_message = f"Spawned Thread {thread.native_id} with {thread_function.__name__} - {name}"

  logger.debug(log_message)

  return thread


def _spawn_main(output_queue, process_function, args, kwargs): #pragma: no cover
  """This function is called from spawn process, it sets up the messaging be

  Args:
      output_queue (_type_): _description_
      process_function (_type_): _description_
      args (_type_): _description_
      kwargs (_type_): _description_
  """
  stdout_capture = StringIO()
  stderr_capture = StringIO()

  if output_queue is not None:
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

  try:
    process_function(*args, **kwargs)
  finally:
    if output_queue is not None:
      output_queue.put(ProgramOutput(
        stdout_capture.getvalue(),
        stderr_capture.getvalue()
      ))

def spawn_process(process_function, *args, start=True, name=None, context=None, capture_output=None, **kwargs):
  logger = kwargs['logger'] if 'logger' in kwargs else getLogger(DICOMNODE_LOGGER_NAME)

  if context is None:
    context = multiprocessing.get_context('spawn')

  if capture_output is None:
    output_queue = None
  else:
    output_queue: Optional[multiprocessing.Queue[ProgramOutput]] = context.Queue()

  process = context.Process(
    target=_spawn_main, args=(output_queue, process_function, args, kwargs ), name=name
  )

  if start:
    process.start()

  log_message = f"Spawned Process {process.pid} with {process_function.__name__}"

  logger.debug(log_message)

  return process, output_queue


class Parallel:
  primitive: ProcessLikeThread | Process
  def __init__(self, primitive: ParallelPrimitive, target, *args, capture_output=None, **kwargs) -> None:
    self.queue = None
    self._program_output: Optional[ProgramOutput] = None
    if primitive == ParallelPrimitive.PROCESS:
      primitive_, queue = spawn_process(target, *args, capture_output=capture_output, **kwargs, context=MULTIPROCESSING)
      self.primitive = primitive_
      self.queue = queue
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
    if self.primitive.is_alive(): # pragma: no cover
      self.join()

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
      case _:     # pragma: no cover
        logger = getLogger(DICOMNODE_LOGGER_NAME)
        logger.critical(f"The Primitive is not a thread or Process")
        return "Unknown"

  def get_output(self) -> ProgramOutput:
    if self.queue is None:
      return ProgramOutput("", "")
    elif self._program_output is None:
      self.join()
      self._program_output = self.queue.get()
      self.queue.close()
    return self._program_output
