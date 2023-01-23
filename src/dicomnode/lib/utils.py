from argparse import ArgumentTypeError
from typing import Any, Callable, Union
from pydicom import Dataset
from typing import Iterable, Mapping, Optional, Union

from threading import Thread

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
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)


    def join(self, *args):
        Thread.join(self, *args)
        return self._return
