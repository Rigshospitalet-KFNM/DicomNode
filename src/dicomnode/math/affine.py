"""Affine Matrixes is describe a the pixel space of an image


"""

# Python standard library
from enum import Enum
from typing import Optional, List, Literal, Tuple, TypeAlias

# Third party packages
from nibabel.nifti1 import Nifti1Image
import numpy
from numpy import array, absolute, dtype, float64, identity, ndarray
from numpy.linalg import inv
from pydicom import Dataset


# Dicomnode packages
from dicomnode.math.types import RotationAxes

RawBasisMatrix: TypeAlias = ndarray[Tuple[Literal[3], Literal[3]], dtype[float64]]


# Rotation matrix can be found here:
# https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
ROTATION_MATRIX_90_DEG_X  = array([
  [1, 0,  0],
  [0, 0, -1],
  [0, 1,  0],
])

ROTATION_MATRIX_90_DEG_Y  = array([
  [ 0, 0, 1],
  [ 0, 1, 0],
  [-1, 0, 0],
])

ROTATION_MATRIX_90_DEG_Z  = array([
  [0, -1, 0],
  [1,  0, 0],
  [0,  0, 1],
])

class Space:
  class Span:
    def __init__(self,
                 x: Tuple[float, float],
                 y: Tuple[float, float],
                 z: Tuple[float, float]) -> None:
      self.x = x
      self.y = y
      self.z = z

    def __getitem__(self, key):
      if key == 1:
        return self.x[0]
      if key == -1:
        return self.x[1]
      if key == 2:
        return self.y[0]
      if key == -2:
        return self.y[1]
      if key == 3:
        return self.z[0]
      if key == -3:
        return self.z[1]

      raise KeyError

  basis: RawBasisMatrix
  span: Span

  def __init__(self, raw: RawBasisMatrix, span: Span):
    self.basis = raw
    self.inverted_raw = inv(raw)
    self.span = span

  @classmethod
  def from_nifti(cls, nifti: Nifti1Image):
    maybe_affine = nifti.affine
    data = nifti.get_fdata()

    if maybe_affine is not None:
      affine = maybe_affine
      span = cls.Span(
        (affine[0,3], 0.0), # wrong values
        (affine[1,3], 0.0), # wrong values
        (affine[2,3], 0.0) # wrong values
      )
    else:
      affine = identity(3, dtype=float64)
      span = cls.Span(
        (0.0, data.shape[2]),
        (0.0, data.shape[1]),
        (0.0, data.shape[0])
      )

    return cls(affine, span)

  @classmethod
  def from_datasets(cls, datasets: List[Dataset]):
    try:
      datasets.sort(key=lambda ds: ds.InstanceNumber)

      first_dataset = datasets[0]
      last_dataset = datasets[-1]
      start_coordinates = first_dataset.ImagePositionPatient
      image_orientation = first_dataset.ImageOrientationPatient
      end_coordinates = last_dataset.ImagePositionPatient
      thickness_x = first_dataset.PixelSpacing[0]
      thickness_y = first_dataset.PixelSpacing[1]
      thickness_z = first_dataset.SliceThickness

      affine_raw = numpy.array([
        [thickness_x * image_orientation[0], thickness_y * image_orientation[3], 0],
        [thickness_x * image_orientation[1], thickness_y * image_orientation[4], 0],
        [thickness_x * image_orientation[2], thickness_y * image_orientation[5], thickness_z, ],
      ])

      end_coordinates = [
        thickness_x * image_orientation[0] * first_dataset.Columns + start_coordinates[0],
        thickness_y * image_orientation[4] * first_dataset.Rows + start_coordinates[1],
        thickness_z * len(datasets) + start_coordinates[2]
      ]

      return cls(
        affine_raw,
        cls.Span(
          (start_coordinates[0], end_coordinates[0]),
          (start_coordinates[1], end_coordinates[1]),
          (start_coordinates[2], end_coordinates[2]),
        )
      )
    except Exception as E:
      print(E)

    pivot = datasets[0]
    x_dim = pivot.Columns
    y_dim = pivot.Rows
    z_dim = len(datasets)
    return cls(identity(3),
               cls.Span((0.0, x_dim),
                        (0.0, y_dim),
                        (0.0, z_dim))
               )


  def correct_rotation(self) -> bool:
    """Detect if the image is rotated such that the reference space makes sense


    Returns:
        bool: _description_
    """

    abs_affine = absolute(self.basis)

    return abs_affine[0,1] < abs_affine[0,0] and \
           abs_affine[0,2] < abs_affine[0,0] and \
           abs_affine[1,0] < abs_affine[1,1] and \
           abs_affine[1,2] < abs_affine[1,1] and \
           abs_affine[2,0] < abs_affine[2,2] and \
           abs_affine[2,1] < abs_affine[2,2]


  def __dominant_axis(self, vector):
    abs_vec = absolute(vector)
    if abs_vec[0] < abs_vec[1]:
      if abs_vec[1] < abs_vec[2]:
        return 2
      else:
        return 1
    else:
      if abs_vec[0] < abs_vec[2]:
        return 2
      else:
        return 0

  def rotation_to_standard_space(self) -> ndarray:
    axis_x_dom = self.__dominant_axis(self.basis[0])
    axis_y_dom = self.__dominant_axis(self.basis[1])
    axis_z_dom = self.__dominant_axis(self.basis[2])

    if axis_x_dom == axis_y_dom or axis_x_dom == axis_z_dom or axis_y_dom == axis_z_dom:
      raise Exception

    if axis_x_dom == 0:
      if axis_y_dom == 1:
        return identity(3)
      else:
        return ROTATION_MATRIX_90_DEG_X
    elif axis_x_dom == 1:
      if axis_y_dom == 0:
        return ROTATION_MATRIX_90_DEG_Z
      else:
        return ROTATION_MATRIX_90_DEG_Z @ ROTATION_MATRIX_90_DEG_X
    else: # axis_x_dom == 2
      if axis_y_dom == 0:
        return ROTATION_MATRIX_90_DEG_X @ ROTATION_MATRIX_90_DEG_Z
      else:
        return ROTATION_MATRIX_90_DEG_Y

  def rotate(self, rotation_matrix):
    pass


  def __matmul__(self, other):
    return self.basis @ other

  def __rmatmul__(self, other):
    return other @ self.basis

class ReferenceSpace(Enum):
  """These are the possible reference spaces a study can be in.

  Here we use the terms defined in:
  https://nipy.org/nibabel/coordinate_systems.html
  """

  RAS = 0
  """Indicate that the image have an affine matrix on the form:
  | x, 0, 0 |
  | 0, y, 0 |
  | 0, 0, z |
  """

  RAI = 1
  """Indicate that the image have an affine matrix on the form:
  | x, 0,  0 |
  | 0, y,  0 |
  | 0, 0, -z |
  """
  RPS = 2
  """Indicate that the image have an affine matrix on the form:
  | x,  0, 0 |
  | 0, -y, 0 |
  | 0,  0, z |
  """
  RPI = 3
  """Indicate that the image have an affine matrix on the form:
  | x, 0,  0 |
  | 0, -y, 0 |
  | 0, 0, -z |
  """
  LAS = 4
  """Indicate that the image have an affine matrix on the form:
  | -x, 0, 0 |
  | 0, y, 0 |
  | 0, 0, z |
  """
  LAI = 5
  """Indicate that the image have an affine matrix on the form:
  | -x, 0, 0 |
  | 0, y, 0 |
  | 0, 0, -z |
  """
  LPS = 6
  """Indicate that the image have an affine matrix on the form:
  | -x, 0, 0 |
  | 0, -y, 0 |
  | 0, 0, z |
  """
  LPI = 7
  """Indicate that the image have an affine matrix on the form:
  | -x, 0, 0 |
  | 0, -y, 0 |
  | 0, 0, -z |
  """

  @classmethod
  def from_affine(cls, affine: Space):
    if not affine.correct_rotation():
      return None

    def is_positive(num):
      return 0 < num

    x_coord = affine.basis[0,0]
    y_coord = affine.basis[1,1]
    z_coord = affine.basis[2,2]

    if is_positive(x_coord) and is_positive(y_coord) and is_positive(z_coord):
      return cls.RAS
    if is_positive(x_coord) and is_positive(y_coord) and not is_positive(z_coord):
      return cls.RAI
    if is_positive(x_coord) and not is_positive(y_coord) and is_positive(z_coord):
      return cls.RPS
    if is_positive(x_coord) and not is_positive(y_coord) and not is_positive(z_coord):
      return cls.RPI
    if not is_positive(x_coord) and is_positive(y_coord) and is_positive(z_coord):
      return cls.LAS
    if not is_positive(x_coord) and is_positive(y_coord) and not is_positive(z_coord):
      return cls.LAI
    if not is_positive(x_coord) and not is_positive(y_coord) and is_positive(z_coord):
      return cls.LPS
    if not is_positive(x_coord) and not is_positive(y_coord) and not is_positive(z_coord):
      return cls.LPI

    return None
