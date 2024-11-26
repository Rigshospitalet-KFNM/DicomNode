"""This module is a wrapper for the low level C++ and CUDA that can increase the
performance of your pipeline.

It's important that you use this module as an API instead of direct calls.
The reason is that Cuda might not be installed, and if you make direct call you
will get some nasty errors.

This module detects if cuda is installed correctly, and if it is it will call
that for you.
  """

# Python standard library
from typing import List, Tuple

# Third party packages
import numpy

# Imports
from dicomnode.math.types import MirrorDirection, CudaErrorEnum, CudaException

# Module Imports
from . import types

# Cuda code
try:
  from dicomnode.math import _cuda

  cudaError, device_prop = _cuda.get_device_properties()
  if not cudaError:
    CUDA = True
  else:
    CUDA = False

except ImportError:
  CUDA = False

from . import space
from . import image

def sum_1_to_n(n: int) -> int:
  return n * (n + 1) // 2

def sum_n_to_m(n:int, m:int):
  return sum_1_to_n(m) - sum_1_to_n(n)

def switch_ordering(input_data: numpy.ndarray) -> numpy.ndarray:
  """Creates an index over the data in another ordering

  See https://en.wikipedia.org/wiki/Row-_and_column-major_order

  Note that

  Args:
    input_data (numpy.ndarray):

  Returns:
    (numpy.ndarray): A view of the data
  """

  return input_data.transpose(tuple(reversed(range(0, input_data.ndim))))

def center_of_gravity(data: numpy.ndarray):
    ndims = data.ndim
    imadim = data.shape
    cog = numpy.zeros((ndims,), dtype=int)

    # loop over each dimension to find center
    for dim in range(0,ndims):
        cog[dim] = numpy.sum(range(0,imadim[dim])*numpy.sum(data, axis=dim))//numpy.sum(numpy.sum(data, axis=dim))

    return cog



def mirror_inplace_gpu(arr: image.numpy_image, direction: MirrorDirection):
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


def mirror(arr: image.numpy_image, direction: MirrorDirection) -> image.numpy_image:
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
    return numpy.flip(arr, 2)
  if direction == MirrorDirection.Y:
    return numpy.flip(arr, 1)
  if direction == MirrorDirection.Z:
    return numpy.flip(arr, 0)
  if direction == MirrorDirection.XY:
    return numpy.flip(arr, (2,1))
  if direction == MirrorDirection.XZ:
    return numpy.flip(arr, (2,0))
  if direction == MirrorDirection.YZ:
    return numpy.flip(arr, (0,1))
  else:
    return numpy.flip(arr, (0,1,2))

def _bounding_box_cpu(array : numpy.ndarray):
  bounding_box_list = [
    [shape_dim - 1, 0] for shape_dim in array.shape
  ]

  for flat_index, value in enumerate(array.flat):
    if value:
      dim_iter = 1
      for shape_index, dim in enumerate(reversed(array.shape)):
        dim_index = (flat_index % (dim * dim_iter)) // dim_iter
        dim_iter *= dim
        current_min = bounding_box_list[shape_index][0]
        current_max = bounding_box_list[shape_index][1]
        bounding_box_list[shape_index][0] = min(current_min, dim_index)
        bounding_box_list[shape_index][1] = max(current_max, dim_index)
  return bounding_box_list

def _bounding_box_gpu(array):
  if not CUDA:
    raise Exception("GPU code call without working GPU driver")

  error, (x_min, x_max, y_min, y_max, z_min, z_max) = _cuda.bounding_box(array)
  if error:
    raise CudaException(CudaErrorEnum(int(error)))

  return (x_min, x_max), (y_min, y_max), (z_min, z_max)

def bounding_box(array):
  if CUDA:
    return _bounding_box_gpu(array)
  else:
    return _bounding_box_cpu(array)

def __all__():
  return [
    space,
    CUDA,
    image,
    mirror,
    types,
    bounding_box,
  ]
