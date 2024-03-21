"""This module is a wrapper for the low level C++ and CUDA that can increase the
performance of your pipeline.

It's important that you use this module as an API instead of direct calls.
The reason is that Cuda might not be installed, and if you make direct call you
will get some nasty errors.

This module detects if cuda is installed correctly, and if it is it will call
that for you.
  """

# Python standard library

# Third party packages
from numpy import flip, rot90

# Imports
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

from . import affine
from . import image


def mirror_inplace_gpu(arr: image.image_type, direction: MirrorDirection):
  """This mirrors the data inplace with a direction.

  Note that this is slower than mirror because it just provides a view of the
  data, rather than doing any actual work.


  Args:
      arr (image.image_type): _description_
      direction (MirrorDirection): _description_

  Raises:
      CudaException: _description_
  """
  # Cuda functions are inplace
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


def mirror(arr: image.image_type, direction: MirrorDirection) -> image.image_type:
  """Provides a view of the data mirrored with respect to the input

  Args:
      arr (image.image_type): This is the image data that needs to be mirrored
      direction (MirrorDirection): This is the direction 

  Raises:
      ValueError: Raised if the input array is not an image

  Returns:
      _type_: _description_
  """
  if len(arr.shape) != 3:
    raise ValueError("Mirror is only supported for 3 dimensional volumes")
  if direction == MirrorDirection.X:
    return flip(arr, 2)
  if direction == MirrorDirection.Y:
    return flip(arr, 1)
  if direction == MirrorDirection.Z:
    return flip(arr, 0)
  if direction == MirrorDirection.XY:
    return flip(arr, (2,1))
  if direction == MirrorDirection.XZ:
    return flip(arr, (2,0))
  if direction == MirrorDirection.YZ:
    return flip(arr, (0,1))
  else:
    return flip(arr, (0,1,2))


def __all__():
  return [
    affine,
    CUDA,
    image,
    mirror,
    types,
  ]
