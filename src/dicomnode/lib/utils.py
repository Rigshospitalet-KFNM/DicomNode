"""Functions that doesn't have any strong home place and no dependencies"""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard Library
from argparse import ArgumentTypeError
import multiprocessing
from logging import Logger, getLogger
import inspect
import sys
import io
from enum import Enum
from contextlib import contextmanager
from threading import Thread
from typing import Any, Optional, Type, Union

# Third party packages
from psutil import Process, STATUS_RUNNING, STATUS_DEAD, STATUS_DISK_SLEEP, STATUS_IDLE, STATUS_LOCKED, STATUS_PARKED, STATUS_SLEEPING, STATUS_STOPPED, STATUS_TRACING_STOP, STATUS_WAITING, STATUS_WAKING, STATUS_ZOMBIE

try:
  from os import getuid, setuid
  UNIX = True
except ImportError: #pragma: no cover
  UNIX = False

from warnings import warn

# Third party packages

# Dicomnode Packages
# This module is imported first, therefore DO NOT PLACE ANY DICOMNODE MODULES IN HERE

# End of imports


def str2bool(v: Union[str, bool]) -> bool:
  """This function convert commons strings to their respective boolean values

  Args:
      v (str): String to be parsed

  Raises:
      ArgumentTypeError: Raised if string is unable to be parsed

  Returns:
      bool: Parsed value
  """
  if isinstance(v, bool):
    return v
  if v.lower() in ['yes', 'true', 't', 'y', '1']:
    return True
  elif v.lower() in ['no', 'false', 'no', 'n','f', '0']:
    return False
  else:
    raise ArgumentTypeError("Boolean value expected")


def prefixInt(number: int, minLength: int= 4):
  numberStr = str(number)
  zeroes = (minLength - len(numberStr)) * "0"
  return f"{zeroes}{numberStr}"


class ThreadWithReturnValue(Thread):
  def __init__(self, group=None, target=None, name=None,
              args=(), kwargs={}, daemon=True):
    Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
    self._return: Any = None

  def run(self):
    if self._target is not None: # type: ignore # In python you don't have private variables
      self._return = self._target(*self._args, **self._kwargs) #type: ignore


  def join(self, *args): #type: ignore
    Thread.join(self, *args)
    return self._return

def drop_privileges(new_user_uid, logger: Optional[Logger] = None, root_uid = 0) -> None: #pragma: no cover
  """Drops privileges of program to run as a user

  An issue with this is that you have to open the socket / files and then drop
  the privileges and sadly you need to go deep in the pynetdicom library.

  Args:
  """
  from os import getuid, setuid
  if getuid() == root_uid:
    if logger is not None:
      logger.info("Dropping privileges to:")
    setuid(new_user_uid)
  else:
    if logger is not None:
      if UNIX:
        logger.info("Cannot drop privileges, not root UID")
      else:
        logger.info("Cannot drop privileges, Not on a unix system")

def deprecation_message(deprecated_module_path, new_module_path) -> None: #pragma: no cover
  warn(f"{deprecated_module_path} has been moved to {new_module_path}", DeprecationWarning)

def type_corrosion(*types: Type):
  """This decorator doesn't really work with class methods, because the self
  arg is not implicitly added. I guess you could figure this out from the
  inspect module.
  """
  def decorator(func):
    def wrapper(*args, **kwargs):
      if len(args) != len(types):
        raise TypeError(f"{func.__name__} expected {len(types)} arguments, but received {len(args)} arguments.")
      new_args = []
      for type_, arg in zip(types, args):
        if isinstance(arg, type_):
          new_args.append(arg)
        else:
          new_args.append(type_(arg))
      return func(*new_args, **kwargs)
    return wrapper
  return decorator

def human_readable_byte_count(number_of_bytes: int):
  """
    Convert bytes to human-readable format (kB, MB, GB, TB).

    Args:
        bytes_number (int): Number of bytes to convert

    Returns:
        str: Converted size with appropriate unit
  """

  units = [
    (1024 ** 4, 'TB'),  # Terabytes
    (1024** 3, 'GB'),  # Gigabytes
    (1024 ** 2, 'MB'),     # Megabytes
    (1024, 'kB')         # Kilobytes
  ]

  if number_of_bytes <= 0:
    return "0 KB"

  # Find the appropriate unit
  for factor, unit in units:
    if factor <= number_of_bytes:
      converted = number_of_bytes / factor
      return f"{converted:.2f} {unit}"

  # If less than 1 KB
  return f"{number_of_bytes} bytes"

def spawn_thread(thread_function, *args, name=None, **kwargs):
  logger = kwargs['logger'] if 'logger' in kwargs else getLogger()

  thread = Thread(
    target=thread_function, args=args, name=name
  )

  thread.start()

  log_message = f"Spawned Thread {thread.native_id} with {thread_function.__name__} - {name}"

  logger.debug(log_message)

  return thread

def spawn_process(process_function, *args, start=True,name=None, **kwargs):
  logger = kwargs['logger'] if 'logger' in kwargs else getLogger()

  #multiprocessing_context = multiprocessing.get_context('spawn')

  process = multiprocessing.Process(
    target=process_function, args=args, name=name
  )

  if start:
    process.start()

  log_message = f"Spawned Process {process.pid} with {process_function.__name__}"

  logger.debug(log_message)

  return process

def name(obj) -> str:
  """Returns the name of a class"""
  return obj.__class__.__name__


class ProcessStatus(Enum):
  RUNNING = STATUS_RUNNING
  DEAD = STATUS_DEAD
  DISK_SLEEP = STATUS_DISK_SLEEP
  IDLE = STATUS_IDLE
  LOCKED = STATUS_LOCKED
  PARKED = STATUS_PARKED
  SLEEPING = STATUS_SLEEPING
  STOPPED = STATUS_STOPPED
  TRACING_STOP = STATUS_TRACING_STOP
  WAITING = STATUS_WAITING
  WAKING = STATUS_WAKING
  ZOMBIE = STATUS_ZOMBIE

def process_status(process: Process):
  return ProcessStatus(process.status())
