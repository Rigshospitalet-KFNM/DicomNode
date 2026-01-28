"""This modules is for creating new threads and processes inside of the
library"""

# Python standard library
from enum import Enum
from logging import getLogger
from threading import Thread
import multiprocessing

# Dicomnode modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME



MULTIPROCESSING = multiprocessing.get_context('spawn')
Process = MULTIPROCESSING.Process

class ParallelPrimitive:
  THREAD = 1
  PROCESS = 2

class Parallel:
  primitive: Thread | Process


  def __init__(self, primitive, target) -> None:
    pass

  def join(self):
    pass

  def id(self):
    if isinstance(self.primitive, Process):
      return self.primitive.pid
    elif isinstance(self.primitive, Thread):
      return self.primitive.native_id
    else:
      logger = getLogger(DICOMNODE_LOGGER_NAME)
      logger.critical(f"The Primitive is not a thread or Process")
