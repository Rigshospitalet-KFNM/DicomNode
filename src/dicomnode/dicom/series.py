"""This module contains the class series, which is an representation of a dicom
series. This class aims to provide a unified view of a series.

https://xkcd.com/927/
"""

# Python standard library
from enum import Enum
from typing import Any, List, Literal, Optional,  Tuple, TypeAlias

# Third party packages
from numpy import zeros_like, ndarray, dtype, float64
from pydicom import Dataset
from nibabel import Nifti1Image

# Dicomnode packages
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING, SIGNED_ARRAY_ENCODING

AffineMatrix: TypeAlias = ndarray[Tuple[Literal[4], Literal[4]], dtype[float64]]

def build_image_from_datasets():
  pass


def scale_image(
                  image: ndarray,
                  bits_stored = 16,
                  bits_allocated = 16,
                ) -> Tuple[ndarray, float, float]:
    target_datatype = UNSIGNED_ARRAY_ENCODING.get(bits_allocated, None)
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
      return zeros_like(image), 1.0, min_val

    image_max_value = ((1 << bits_stored) - 1)

    slope = (max_val - min_val) / image_max_value
    intercept = min_val

    new_image = ((image - intercept) / slope).astype(target_datatype)

    return new_image, slope, intercept



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

SERIES_VARYING_TAGS = set([
  0x00080018, # SOPInstanceUID
  0x00200013, # InstanceNumber
  0x00280106, # SmallestImagePixelValue
  0x00280106, # LargestImagePixelValue
  0x00281052, # RescaleIntercept
  0x00281053, # RescaleSlope
  0x73E00010, # PixelData
])
"""This is a list of tags that are assumed to be varying across a series
Note that this is not a complete list. There's for instance dicom series where
the slice length is difference per slice, however this library doesn't support
this.
"""

class Series:
  """

  Returns:
      _type_: _description_
  """

  uid = None
  datasets: Optional[List[Dataset]] = None
  nifti: Optional[Nifti1Image] = None
  image_data: Optional[ndarray] = None
  affine: Optional[AffineMatrix] = None
  reference_space = None

  # Constructors
  def __init__(self) -> None:
    pass

  @classmethod
  def from_dicom(cls, datasets: List[Dataset]):
    series = cls()
    if 0 < len(datasets):
      series.datasets = datasets

    return series

  @classmethod
  def from_nifti(cls, nifti: Nifti1Image):
    series = cls()
    series.nifti = nifti

    return series

  def __getitem__(self, tag):
    if self.datasets is None:
      return None

    if tag in SERIES_VARYING_TAGS:
      return [dataset.get(tag, None) for dataset in self.datasets]
    return self.datasets[0].get(tag, None)

  def can_copy_into_image(self, image:ndarray[Tuple[int,int,int],Any]) -> bool:
    if self.datasets is None:
      return False
    return image.shape[2] == len(self.datasets)

  def generate_numpy(self) -> ndarray:
    if self.image_data is not None:
      return self.image_data

    if self.nifti is not None:
      self.affine = self.nifti.affine
      if self.affine is not None:
        self.reference_space = detect_reference_space(self.affine)
      self.image_data = self.nifti.get_fdata()
      return self.image_data

    if 



