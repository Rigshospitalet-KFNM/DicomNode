"""This module contains the class series, which is an representation of a dicom
series. This class aims to provide a unified view of a series.

https://xkcd.com/927/
"""

# Python standard library
from functools import reduce
from enum import Enum
from typing import Any, List, Literal, Optional,  Tuple, TypeAlias, Union

# Third party packages
from numpy import zeros_like, ndarray, dtype, float64, float32, empty, absolute
from pydicom import Dataset, DataElement
from pydicom.tag import BaseTag
from nibabel import Nifti1Image

# Dicomnode packages
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING, SIGNED_ARRAY_ENCODING
from dicomnode.math.affine import AffineMatrix, ReferenceSpace, build_affine_from_datasets
from dicomnode.math.image import build_image_from_datasets
from dicomnode.lib.exceptions import InvalidDataset

def sortDatasets(dataset: Dataset):
  """Sorting function for a collection of datasets. The order is determined by
  the instance number

  Args:
      dataset (Dataset): _description_

  Returns:
      int: _description_
  """
  return dataset.InstanceNumber

def shared_tag(datasets: List[Dataset], tag: BaseTag) -> bool:
  """Determines if tag is shared, meaning that for all datasets the value of the
  tag are equal. This is includes if the tag is missing, then that is considered
  a shared tag

  Args:
      datasets (List[Dataset]): A collection of datasets for which the
      "sharedness" is in question
      tag (pydicom.tag.BaseTag): This is tag that you wish determine the
      "sharedness" of

  Raises:
      ValueError: Raised when passed an empty list. Wrap it in a try catch block
      if you have a strong opinion on the result.

  Returns:
      bool: True if the tag is shared, false if not
  """
  if len(datasets) == 0:
    raise ValueError("Cannot determine if a tag is unique from an empty collection")

  # This is many here because and is not a standard python function
  def fun_and(x, y):
    return x and y

  pivot = datasets[0]
  return reduce(fun_and,
                [dataset.get(tag, None) == pivot.get(tag, None)
                  for dataset in datasets],
                True)

SERIES_VARYING_TAGS = set([
  0x00080018, # SOPInstanceUID
  0x00200013, # InstanceNumber
  0x00200032, # ImagePositionPatient
  0x00201041, # SliceLocation
  0x00280106, # SmallestImagePixelValue
  0x00280106, # LargestImagePixelValue
  0x00281052, # RescaleIntercept
  0x00281053, # RescaleSlope
  0x73E00010, # PixelData
])
"""This is a list of tags that are assumed to be varying across a series
Note that this is not a complete list.
There is valid dicom series that have more varying tags than these, however this
library doesn't support those.
"""

class Series:
  """Base class for a Collection of images, that together form a series or a
  cubic volume that contains an tomographic image.

  It's a
  """

  image_data: ndarray
  affine: Optional[AffineMatrix] = None
  reference_space: Optional[ReferenceSpace] = None

  # Constructors
  def __init__(self,
               image_data: ndarray[Tuple[int,int,int], Any],
               affine:Optional[AffineMatrix]) -> None:
    self.image_data = image_data
    self.affine = affine
    if self.affine is not None:
      self.reference_space = ReferenceSpace.detect_reference_space(self.affine)

class DicomSeries(Series):
  pivot: Dataset
  datasets: List[Dataset]

  def __init__(self, datasets: List[Dataset]) -> None:
    if len(datasets) == 0:
      raise ValueError("Cannot construct a dicom series from an empty list")

    datasets.sort(key=sortDatasets)
    self.datasets = datasets
    self.pivot = self.datasets[0]

    image = build_image_from_datasets(self.datasets)
    # It's forbidden to call method on self, since the object have not been
    # Constructed yet!
    if shared_tag(datasets, 0x00180050):
      affine = build_affine_from_datasets(self.pivot)
    else:
      affine = None

    super().__init__(image, affine)

  def __getitem__(self, tag) -> Optional[Union[DataElement, List[DataElement]]]:
    if self.datasets is None:
      return None

    if tag in SERIES_VARYING_TAGS:
      return [dataset.get(tag, None) for dataset in self.datasets]
    return self.pivot.get(tag, None)

  def set_shared_tag(self, tag: BaseTag, value: DataElement):
    for dataset in self.datasets:
      dataset[tag] = value

  def set_individual_tag(self, tag: BaseTag, values: List[DataElement]):
    if len(values) == len(self.datasets):
      raise ValueError("The amount of values doesn't match the amount datasets")
    for dataset, value in zip(self.datasets, values):
      dataset[tag] = value

  def can_copy_into_image(self, image:ndarray[Tuple[int,int,int],Any]) -> bool:
    return image.shape[2] == len(self.datasets)

  def shared_tag(self, tag) -> bool:
    return shared_tag(self.datasets, tag)

class NiftiSeries(Series):
  def __init__(self, nifti: Nifti1Image) -> None:
    self.nifti = nifti
    image_data = self.nifti.get_fdata()

    super().__init__(image_data, self.nifti.affine)
