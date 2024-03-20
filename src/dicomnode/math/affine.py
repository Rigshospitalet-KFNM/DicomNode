"""Affine Matrixes is describe a the pixel space of an image


"""

# Python standard library
from enum import Enum
from typing import Optional, List, Literal, Tuple, TypeAlias

# Third party packages
from numpy import absolute, dtype, float64, ndarray
from pydicom import Dataset

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

  @classmethod
  def correct_rotation(cls, affine: AffineMatrix) -> bool:
    """Detect if the image is rotated such that the reference space makes sense

    Args:
        affine (AffineMatrix): _description_

    Returns:
        bool: _description_
    """
    abs_affine = absolute(affine)

    return abs_affine[0,1] < abs_affine[0,0] and \
           abs_affine[0,2] < abs_affine[0,0] and \
           abs_affine[1,0] < abs_affine[1,1] and \
           abs_affine[1,2] < abs_affine[1,1] and \
           abs_affine[2,0] < abs_affine[2,2] and \
           abs_affine[2,1] < abs_affine[2,2]

  @classmethod
  def from_affine(cls, affine: AffineMatrix):
    if not cls.correct_rotation(affine):
      return None

    def is_positive(num):
      return 0 < num

    x_coord = affine[0,0]
    y_coord = affine[1,1]
    z_coord = affine[2,2]

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

def build_affine_from_datasets(datasets: List[Dataset]) -> Optional[AffineMatrix]:
  try:
    datasets.sort(key=lambda ds: ds.InstanceNumber)

    first_dataset = datasets[0]
    last_dataset = datasets[-1]
    start_coordinates = first_dataset.ImagePositionPatient
  


  except ValueError:
    return None
  except AttributeError:
    return None


