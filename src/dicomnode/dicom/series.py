"""This module contains the class series, which is an representation of a dicom
series. This class aims to provide a unified view of a series.

https://xkcd.com/927/
"""

# Python standard library
from functools import reduce
from enum import Enum
from typing import Any, Callable, List, Literal, Optional,  Tuple, TypeAlias, Union

# Third party packages
from numpy import zeros_like, ndarray, dtype, float64, float32, empty, absolute
from pydicom import Dataset, DataElement
from pydicom.datadict import dictionary_VR
from pydicom.tag import BaseTag
from nibabel.nifti1 import Nifti1Image

# Dicomnode packages
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING, SIGNED_ARRAY_ENCODING
from dicomnode.math.affine import AffineMatrix, ReferenceSpace
from dicomnode.math.image import build_image_from_datasets, numpy_image, Image
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

  @property
  def image(self):
    if self._image is None:
      if self._image_constructor is None: #pragma: no cover
        raise Exception
      self._image = self._image_constructor()
    return self._image

  # Constructors
  def __init__(self, image: Union[Image, Callable[[],Image]]):
    if isinstance(image, Image):
      self._image = image
      self._image_constructor = None
    else:
      self._image = None
      self._image_constructor = image

class DicomSeries(Series):
  pivot: Dataset
  datasets: List[Dataset]

  def __init__(self, datasets: List[Dataset]) -> None:
    if len(datasets) == 0:
      raise ValueError("Cannot construct a dicom series from an empty list")

    datasets.sort(key=sortDatasets)
    self.datasets = datasets
    self.pivot = self.datasets[0]

    def image_constructor():
      return Image.from_datasets(self.datasets)

    # It's forbidden to call method on self, since the object have not been
    # Constructed yet!
    super().__init__(image_constructor)

  def __iter__(self):
    for dataset in self.datasets:
      yield dataset

  def __len__(self):
    return len(self.datasets)

  def __getitem__(self, tag) -> Optional[Union[DataElement, List[DataElement]]]:
    if tag in SERIES_VARYING_TAGS:
      return [dataset.get(tag, None) for dataset in self.datasets]
    return self.pivot.get(tag, None)

  def __setitem__(self, tag: int, value):
    if tag in SERIES_VARYING_TAGS:
      if not isinstance(value, List):
        error_message = f"The tag is a varying dicom tag. The correct type is a list of length {len(self)}"
        raise TypeError(error_message)
      self.set_individual_tag(tag, value)
    else:
      if not isinstance(value, DataElement):
        value = DataElement(tag, dictionary_VR(tag), value)

      self.set_shared_tag(tag, value)

  def set_shared_tag(self, tag: int, value: DataElement):
    for dataset in self.datasets:
      dataset[tag] = value

  def set_individual_tag(self, tag: int, values: List[DataElement]):
    if len(values) != len(self):
      error_message = f"The amount of values ({len(values)}) doesn't match the amount datasets ({len(self)})"
      raise ValueError(error_message)
    for dataset, value in zip(self.datasets, values):
      if not isinstance(value, DataElement):
        value = DataElement(tag, dictionary_VR(tag), value)
      dataset[tag] = value

  def can_copy_into_image(self, image:ndarray[Tuple[int,int,int],Any]) -> bool:
    return image.shape[2] == len(self.datasets)

  def shared_tag(self, tag) -> bool:
    return shared_tag(self.datasets, tag)

class NiftiSeries(Series):
  def __init__(self, nifti: Nifti1Image) -> None:
    self.nifti = nifti
    image_data = self.nifti.get_fdata()
    affine = AffineMatrix.from_nifti(self.nifti)

    super().__init__(Image(image_data, affine))

__all__ = [
  'Series',
  'DicomSeries',
  'NiftiSeries',
]
