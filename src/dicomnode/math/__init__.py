"""This module is a wrapper for the low level C++ and CUDA that can increase the
performance of your pipeline.

It's important that you use this module as an API instead of direct calls.
The reason is that Cuda might not be installed, and if you make direct call you
will get some nasty errors.

This module detects if cuda is installed correctly, and if it is it will call
that for you.
  """

# Python standard library
from typing import Tuple

# Third party packages
from numpy import ndarray, zeros_like

# Imports
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING
from dicomnode.math.types import MirrorDirection, CudaErrorEnum, CudaException


# Module Imports
from . import types


# Cpp code
from . import _c # type: ignore

# Cuda code
try:
  from . import _cuda # type: ignore
  CUDA = True
except ImportError:
  CUDA = False

def mirror(arr: ndarray, direction: MirrorDirection):
  if len(arr.shape) == 3:
    raise ValueError("Mirror is only supported for 3 dimensional volumes")

  if CUDA:
    if direction == MirrorDirection.X:
      func = _cuda.mirror_x
    elif direction == MirrorDirection.Y:
      func = _cuda.mirror_y
    elif direction == MirrorDirection.Z:
      func = _cuda.mirror_z
    elif direction == MirrorDirection.XY:
      func = _cuda.mirror_xy
    elif direction == MirrorDirection.XZ:
      func = _cuda.mirror_xz
    elif direction == MirrorDirection.YZ:
      func = _cuda.mirror_yz
    else:
      func = _cuda.mirror_xyz

    error = CudaErrorEnum(func(arr))

    if error != CudaErrorEnum.cudaSuccess:
      raise CudaException(error)
  else:
    pass

from . import affine
from . import image

def __all__():
  return [
    affine,
    image,
    mirror,
    types,
  ]
