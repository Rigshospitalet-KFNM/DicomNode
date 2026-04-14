"""This module contain various function, helping with input validation

The goal is that the cuda code doesn't need to practice defensive coding,
such that the cuda code can be simpler, since dealing with problems is much
easier in python than cuda-cpp.

In other words it's this modules function responsibility to ensure that the cuda
function CANNOT FAIL.
"""
# Python Standard library
from typing import Any, Tuple

from dicomnode.lib.exceptions import GPUError, ContractViolation

def gpu_call(func, *args, **kwargs) -> Any:
  from dicomnode.math import _cuda
  return_value = func(*args, **kwargs)

  if isinstance(return_value, _cuda.DicomnodeError) and return_value:
    raise GPUError(f"Encountered {return_value}")

  if isinstance(return_value, tuple):
    error = return_value[0]

    if not isinstance(error, _cuda.DicomnodeError):
      raise ContractViolation("A GPU didn't return a GPU error flag to indicate success")

    if error:
      raise GPUError(f"Encountered {error}")

    return return_value[1:]
