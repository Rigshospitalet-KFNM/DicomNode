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
from dicomnode.dicom.series import extract_image, ImageContainerType
from dicomnode.math import CUDA, switch_ordering
from dicomnode.math.space import Space
from dicomnode.math.image import Image

if CUDA:
  from dicomnode.math import _cuda

class RESAMPLE_METHODS(Enum):
  LINEAR = "linear"

def resample(source: ImageContainerType,
             target: Union[Space, ImageContainerType],
             method=RESAMPLE_METHODS.LINEAR
             ):
  source = extract_image(source)

  if not isinstance(target, Space):
    target = extract_image(target).space

  if CUDA:
    success, interpolated =  _cuda.interpolation.linear(source, target)

    return interpolated
  else:
    return cpu_interpolate(source, target, method)


def cpu_interpolate(source: Image, target: Space, method=RESAMPLE_METHODS.LINEAR):
  original_grid = [numpy.arange(s) for s in reversed(source.space.domain)]

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
  interpolated: numpy.ndarray = interpolator(orig_indices).reshape(target.domain) # type: ignore

  return interpolated
