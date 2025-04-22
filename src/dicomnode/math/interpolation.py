"""This module handles interpolation / resampling of dicom images
It's mostly a wrapper around scipy / calling the cuda functions

"""

# TODO: make this module load lazy because it depend on scipy

# Python standard library
from enum import Enum
from typing import Union

# Third party modules
import numpy
from scipy.interpolate import RegularGridInterpolator

# Dicomnode modules
from dicomnode.lib.logging import get_logger
from dicomnode.dicom.series import extract_image, ImageContainerType, extract_space
from dicomnode.math import CUDA, switch_ordering
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
    success, interpolated =  _cuda.interpolation.linear(source, target)

    return Image(interpolated, target)
  else:
    return cpu_interpolate(source, target, method) # pragma: no cover # I test on gpu devices


def cpu_interpolate(source: Image, target: Space, method=RESAMPLE_METHODS.LINEAR):
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
  original_grid = [numpy.arange(s) for s in reversed(source.space.extent)]

  # Create interpolator for original data
  interpolator = RegularGridInterpolator(
      tuple(original_grid),
      switch_ordering(source.raw),
      method=method.value,
      bounds_error=False,
      fill_value=source.minimum_value
  )

  # Create new grid coordinates
  new_coords = numpy.array([i for i in target.coords()])

  world_coords_new = target.starting_point + new_coords @ target.basis

  # Transform world coordinates back to original basis indices for interpolation
  orig_indices = (world_coords_new - source.space.starting_point) @ source.space.inverted_basis

  # Interpolate
  interpolated: numpy.ndarray = interpolator(orig_indices).reshape(target.extent) # type: ignore

  return Image(interpolated, target)
