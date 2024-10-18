"""Functions that doesn't have any strong home place

"""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard Library
from argparse import ArgumentTypeError
from threading import Thread
from logging import Logger
from typing import Any, Optional, Tuple, Type, TypeVar, Union
try:
  from os import getuid, setuid
  UNIX = True
except ImportError:
  UNIX = False

from warnings import warn

# Third party packages
import numpy


# Dicomnode Packages
# This module is imported first, therefore DO NOT PLACE ANY DICOMNODE MODULES IN HERE

# End of imports


def str2bool(v:Union[str, bool]) -> bool:
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


  def join(self, *args):
    Thread.join(self, *args)
    return self._return

def colomn_to_row_major_order(input: numpy.ndarray[Tuple[int,int,int], Any]) -> numpy.ndarray:
  """Converts an array from a Column major to row major

  See https://en.wikipedia.org/wiki/Row-_and_column-major_order

  Args:
    input (numpy.ndarray): An three dimensional array of dimensions (x,y,z)

  Returns:
    (numpy.ndarray): An three dimensional array of dimensions (z,y,x) containing the data
  """
  # I should test this function on some real data.

  if len(input.shape) != 3:
    raise TypeError("Invalid Shape, accepts only three dimensional arrays")

  return_array = numpy.empty(tuple(reversed(input.shape)), order='C')

  for index in range(input.shape[2]):
    return_array[index, :, :] = input[:,:,index].T

  return return_array

def drop_privileges(new_user_uid, logger: Optional[Logger] = None, root_uid = 0) -> None:
  """Drops privileges of program to run as a user

  An issue with this is that you have to open the socket / files and then drop the privileges
  and sadly you need to go deep in the pynetdicom library.

  Args:
  """
  if UNIX and getuid() == root_uid:
    if logger is not None:
      logger.info("Dropping privileges to:")
    setuid(new_user_uid)
  else:
    if logger is not None:
      if UNIX:
        logger.info("Cannot drop privileges, not root UID")
      else:
        logger.info("Cannot drop privileges, Not on a unix system")


def deprecation_message(deprecated_module_path, new_module_path) -> None:
  warn(f"{deprecated_module_path} has been moved to {new_module_path}", DeprecationWarning)

def type_corrosion(*types: Type):
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