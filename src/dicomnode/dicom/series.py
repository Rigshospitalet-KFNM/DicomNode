"""This module contains the class series, which is an representation of a dicom
series. This class aims to provide a unified view of a series.

https://xkcd.com/927/
"""

# Python standard library
from enum import Enum

# Third party packages

# Dicomnode packages

class ReferenceSpace(Enum):
  """These are the possible reference spaces a study can be in. 

  Args:
      Enum (_type_): _description_
  """

  RAS = 0
  RAI = 1
  RPS = 2
  RPI = 3
  RAS = 4
  RAI = 5
  RPS = 6
  RPI = 7


class Series:
  datasets = None
  nifti = None
  numpy = None
  affine = None

  @classmethod from_dicom()

