"""This module contains the class series, which is an representation of a dicom
series. This class aims to provide a unified view of a series.

https://xkcd.com/927/
"""

# Python standard library
from enum import Enum
from typing import List, Literal, Optional,  Tuple, TypeAlias

# Third party packages
from numpy import ndarray, dtype, float64
from pydicom import Dataset
from nibabel import Nifti1Image

# Dicomnode packages


AffineMatrix: TypeAlias = ndarray[Tuple[Literal[4], Literal[4]], dtype[float64]]

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

def is_positive(num):
  return 0 < num

def detect_reference_space(affine: AffineMatrix) -> ReferenceSpace:
  x_coord = affine[0,0]
  y_coord = affine[1,1]
  z_coord = affine[2,2]

  if is_positive(x_coord) and is_positive(y_coord) and is_positive(z_coord):
    return ReferenceSpace.RAS
  if is_positive(x_coord) and is_positive(y_coord) and not is_positive(z_coord):
    return ReferenceSpace.RAI
  if is_positive(x_coord) and not is_positive(y_coord) and is_positive(z_coord):
    return ReferenceSpace.RPS
  if is_positive(x_coord) and not is_positive(y_coord) and not is_positive(z_coord):
    return ReferenceSpace.RPI
  if not is_positive(x_coord) and is_positive(y_coord) and is_positive(z_coord):
    return ReferenceSpace.LAS
  if not is_positive(x_coord) and is_positive(y_coord) and not is_positive(z_coord):
    return ReferenceSpace.LAI
  if not is_positive(x_coord) and not is_positive(y_coord) and is_positive(z_coord):
    return ReferenceSpace.LPS
  if not is_positive(x_coord) and not is_positive(y_coord) and not is_positive(z_coord):
    return ReferenceSpace.LPI


class Series:
  """_summary_

  Returns:
      _type_: _description_
  """

  uid = None
  datasets = None
  nifti = None
  numpy = None
  affine: Optional[AffineMatrix] = None
  reference_space = None

  def __init__(self) -> None:
    pass

  @classmethod
  def from_dicom(cls, datasets: List[Dataset]):
    series = cls()
    series.datasets = datasets

    return series

  def from_nifti(cls, nifti: Nifti1Image):
    series = cls()
    series.nifti = nifti


    return series