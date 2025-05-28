"""This module handles interpolation / resampling of dicom images
It's mostly a wrapper around scipy / calling the cuda functions

"""

# TODO: make this module load lazy because it depend on scipy

# Python standard library
from enum import Enum
from typing import Union

# Third party modules
import numpy

# Dicomnode modules
from dicomnode.lib.logging import get_logger
from dicomnode.dicom.series import extract_image, ImageContainerType, extract_space
from dicomnode.math import CUDA, switch_ordering, _cpp
from dicomnode.math.space import Space
from dicomnode.math.image import Image

class RESAMPLE_METHODS(Enum):
  LINEAR = "linear"

def resample(source: ImageContainerType,
             target: Union[Space, ImageContainerType],
             method= RESAMPLE_METHODS.LINEAR
             ) -> Image:
  """Creates an image with the target space, where the data is interpolated /
  resamples from the source.

  Args:
      source (ImageContainerType): _description_
      target (Union[Space, ImageContainerType]): _description_
      method (RESAMPLE_METHODS, optional): _description_. Defaults to RESAMPLE_METHODS.LINEAR.

  Returns:
      Image: _description_
  """

  source = extract_image(source)

  if not isinstance(target, Space):
    target = extract_space(target)

  if CUDA:
    from dicomnode.math import _cuda
    error, interpolated =  _cuda.interpolation.linear(source, target)

    if error:
      raise Exception(f"Cpp code encountered a lower level exception: {error}")

    return Image(interpolated, target)
  else:
    return cpu_interpolate(source, target)


def cpu_interpolate(source: Image, target: Space):
  """Creates an image with the target space, where the data is interpolated /
  resamples from the source.

  Forces to run the cpu.

  Note that this function

  Args:
      source (Image): _description_
      target (Space): _description_
      method (_type_, optional): _description_. Defaults to RESAMPLE_METHODS.LINEAR.

  Returns:
      _type_: _description_
  """

  success, interpolated = _cpp.interpolation.linear(source, target)

  if str(success) != "Success":
    raise Exception(f"Cpp code encountered a lower level exception: {success}")

  return Image(interpolated, target)
