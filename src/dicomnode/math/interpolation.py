"""This module handles interpolation / resampling of dicom images
It's mostly a wrapper around scipy / calling the cuda functions

"""

# TODO: make this module load lazy because it depend on scipy

# Python standard library
from enum import Enum
from typing import Any, Iterable, List, Literal, Tuple, Union

# Third party modules
import numpy
from scipy.interpolate import RegularGridInterpolator

# Dicomnode modules
from dicomnode.dicom.series import extract_image, ImageContainerType
from dicomnode.math.space import Space
from dicomnode.math.image import Image

try:
  from dicomnode.math import _cuda
  CUDA = True
except ImportError: #pragma: no cover
  CUDA = False #pragma: no cover

class RESAMPLE_METHODS(Enum):
  LINEAR = "linear"

def resample(source: ImageContainerType,
             target: Union[Space, ImageContainerType],
             method=RESAMPLE_METHODS.LINEAR
             ):
  source = extract_image(source)

  if not isinstance(target, Space):
    target = extract_image(target).space

  return cpu_interpolate(source, target, method)
  #if CUDA:
  #  return _cuda.interpolation.linear(source, target)
  #else:


def cpu_interpolate(source: Image, target: Space, method=RESAMPLE_METHODS.LINEAR):
  original_grid = [numpy.arange(s) for s in source.space.domain]

  # Create interpolator for original data
  interpolator = RegularGridInterpolator(
      tuple(original_grid),
      source.raw,
      method=method.value,
      bounds_error=False,
      fill_value=0
  )

  # Create new grid coordinates
  new_grid = [numpy.arange(s) for s in target.domain]
  new_I, new_J, new_K = numpy.meshgrid(*new_grid, indexing='ij')

  # Convert new indices to world coordinates
  new_coords = numpy.stack([new_I, new_J, new_K], axis=-1)
  new_coords = new_coords.reshape(-1, source.raw.ndim)
  world_coords_new = target.starting_point + new_coords @ target.basis

  # Transform world coordinates back to original basis indices for interpolation
  # Solve: world_coords = original_start + coords @ original_basis
  # Therefore: coords = (world_coords - original_start) @ inv(original_basis)
  orig_indices = (world_coords_new - source.space.starting_point) @ source.space.inverted_basis

  # Interpolate
  interpolated: numpy.ndarray = interpolator(orig_indices).reshape(target.domain) # type: ignore

  return interpolated
