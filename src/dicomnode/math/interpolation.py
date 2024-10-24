"""This module handles interpolation / resampling of dicom images
It's mostly a wrapper around scipy / calling the cuda functions

"""

# TODO: make this module load lazy because it depend on scipy

# Python standard library
from typing import Any, Iterable, Literal, Tuple

# Third party modules
import numpy
from scipy.interpolate import RegularGridInterpolator

# Dicomnode modules
from dicomnode.math.space import Space

try:
  from dicomnode.math import _cuda
  CUDA = True
except ImportError:
  CUDA = False

def resample():
  pass


def py_interpolate(data: numpy.ndarray,
                              original_basis,      # 3x3 matrix [v1, v2, v3]
                              original_start,      # (x,y,z) of index (0,0,0)
                              new_basis,           # 3x3 matrix [v1, v2, v3]
                              new_start,           # (x,y,z) of where new grid starts
                              new_shape: Iterable):          # (nx, ny, nz) of output

  # Convert bases to numpy arrays if they aren't already
  original_basis = numpy.array(original_basis)
  new_basis = numpy.array(new_basis)
  original_start = numpy.array(original_start)
  new_start = numpy.array(new_start)

  original_grid = [numpy.arange(s) for s in data.shape]

  # Create interpolator for original data
  interpolator = RegularGridInterpolator(
      tuple(original_grid),
      data,
      method='linear',
      bounds_error=False,
      fill_value=0
  )

  # Create new grid coordinates
  new_grid = [numpy.arange(s) for s in new_shape]
  new_I, new_J, new_K = numpy.meshgrid(*new_grid, indexing='ij')

  # Convert new indices to world coordinates
  new_coords = numpy.stack([new_I, new_J, new_K], axis=-1)
  new_coords = new_coords.reshape(-1, data.ndim)
  world_coords_new = new_start + new_coords @ new_basis

  # Transform world coordinates back to original basis indices for interpolation
  # Solve: world_coords = original_start + coords @ original_basis
  # Therefore: coords = (world_coords - original_start) @ inv(original_basis)
  orig_indices = (world_coords_new - original_start) @ numpy.linalg.inv(original_basis)

  # Interpolate
  interpolated: numpy.ndarray = interpolator(orig_indices).reshape(new_shape) # type: ignore

  return interpolated
