from argparse import ArgumentTypeError
from typing import Any, Callable, Union
from pydicom import Dataset

def str2bool(v:str) -> bool:
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

def prefixInt(number: int, minLength : int= 4):
  numberStr = str(number)
  zeroes = (minLength - len(numberStr)) * "0"
  return f"{zeroes}{numberStr}"

def getTag(Tag : int) -> Callable[[Dataset], Any]:
  def retfunc(Dataset):
    if Tag in Dataset:
      return Dataset[Tag]
    else:
      return None
  return retfunc

