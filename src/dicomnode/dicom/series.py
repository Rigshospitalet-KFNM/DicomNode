"""This module contains the class series, which is an representation of a dicom
series. This class aims to provide a unified view of a series.

https://xkcd.com/927/
"""

# Python standard library
from datetime import datetime
from functools import reduce
from enum import Enum
from typing import Any, Callable, Dict, List, Iterable, Literal, Optional,\
  Tuple, TypeAlias, Union

# Third party packages
import numpy
from numpy import zeros_like, ndarray, dtype, float64, float32, empty, absolute
from pydicom import Dataset, DataElement
from pydicom.datadict import dictionary_VR
from pydicom.tag import BaseTag
from nibabel.nifti1 import Nifti1Image

# Dicomnode packages
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING, SIGNED_ARRAY_ENCODING

from dicomnode.dicom import has_tags
from dicomnode.math.space import Space, ReferenceSpace
from dicomnode.math.image import Image, FramedImage
from dicomnode.lib.exceptions import InvalidDataset, MissingPivotDataset, IncorrectlyConfigured

def sortDatasets(dataset: Dataset):
  """Sorting function for a collection of datasets. The order is determined by
  the instance number

  Args:
      dataset (Dataset): _description_

  Returns:
      int: _description_
  """
  if 'ImagePositionPatient' in dataset:


    return dataset.ImagePositionPatient[2]

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
  0x0008_0018, # SOPInstanceUID
  0x0020_0013, # InstanceNumber
  0x0020_0032, # ImagePositionPatient
  0x0020_1041, # SliceLocation
  0x0028_0106, # SmallestImagePixelValue
  0x0028_0106, # LargestImagePixelValue
  0x0028_1052, # RescaleIntercept
  0x0028_1053, # RescaleSlope
  0x0054_1330, # ImageIndex
  0x73E0_0010, # PixelData
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
      if isinstance(self._image_constructor, Callable): #pragma: no cover
        self._image = self._image_constructor()
      else:
        raise IncorrectlyConfigured("An Image must have an image or ")
    return self._image

  # Constructors
  def __init__(self, image: Union[Image, FramedImage, Callable[[],Union[Image, FramedImage]]]):
    if isinstance(image, Image) or isinstance(image, FramedImage):
      self._image = image
      self._image_constructor = None
    else:
      self._image = None
      self._image_constructor = image

class DicomSeries(Series):
  """This represenst
  """
  pivot: Dataset
  datasets: List[Dataset]

  def __init__(self, datasets: List[Dataset]) -> None:
    if len(datasets) == 0:
      raise ValueError("Cannot construct a dicom series from an empty list")

    self.datasets = datasets
    self.pivot = self.datasets[0]
    if 'InstanceNumber' in self.pivot:
      datasets.sort(key=sortDatasets)
    else:
      self.set_individual_tag(0x0020_0013, [i + 1 for i,_ in enumerate(self.datasets)])

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

  def __getattribute__(self, name: str) -> Any:
    try:
      return super().__getattribute__(name)
    except AttributeError:
        return self.datasets[0].__getattribute__(name)

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

  def set_individual_tag(self, tag: int, values: List[Union[DataElement, Any]]):
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
    affine = Space.from_nifti(self.nifti)

    super().__init__(Image(image_data, affine))

class LargeDynamicPetSeries(Series):
  REQUIRED_TAGS = [
    'NumberOfSlices',
    'NumberOfTimeSlices',
    'Rows',
    'Cols',
    'RescaleIntercept',
    'RescaleSlope',
    'PixelData',
    'ActualFrameTime',
    'AcquisitionTime',
    'AcquisitionDate',
    'ImageIndex',
  ]

  @property
  def image(self) -> FramedImage:
    image = super().image
    if not isinstance(image, FramedImage):
      raise IncorrectlyConfigured

    return image

  @property
  def raw(self):
    return self.image.raw

  @property
  def frame_durations_ms(self):
    return self._frame_durations_ms

  @property
  def frame_acquisition_time(self):
    return self._frame_acquisition_time

  @property
  def pixel_volume(self):
    return self._pixel_volume

  def __init__(self, datasets: Iterable[Dataset]):
    first_dataset = None
    raw_image = None
    frame_times_ms = None
    frame_acquisition_time = None

    def insert_image(raw:ndarray[Tuple[int,int,int,int], Any], dataset: Dataset):
      if not has_tags(dataset, self.REQUIRED_TAGS):
        raise InvalidDataset(f"Dataset doesn't appear to be large pet")

      time_series = dataset.ImageIndex // dataset.NumberOfSlices
      slice_number_in_series = dataset.ImageIndex % dataset.NumberOfSlices

      raw[time_series, slice_number_in_series, :, :] = \
        dataset.pixel_array.astype(float32) * dataset.RescaleSlope\
          + dataset.RescaleIntercept

    for dataset in datasets:
      if first_dataset is None:
        first_dataset = dataset

      if raw_image is None:
        raw_image = empty((dataset.NumberOfTimeSlices, dataset.NumberOfSlices, dataset.Rows, dataset.Cols), float32)

      if frame_times_ms is None:
        frame_times_ms = numpy.ones((dataset.NumberOfTimeSlices), dtype=numpy.int32)

      if frame_acquisition_time is None:
        frame_acquisition_time = numpy.ones((dataset.NumberOfTimeSlices), dtype='datetime64[ms]')

      frameIndex = dataset.ImageIndex // dataset.NumberOfSlices
      insert_image(raw_image, dataset)
      frame_times_ms[frameIndex] = dataset.ActualFrameDuration
      frame_acquisition_time[frameIndex] = datetime.strptime(dataset.AcquisitionDate+dataset.AcquisitionTime, "%Y%m%d%H%M%S.%f")

    if first_dataset is None:
      raise MissingPivotDataset("Cannot construct an image from no datasets")
    if raw_image is None:
      raise MissingPivotDataset("Cannot construct an image from no datasets")
    if frame_times_ms is None:
      raise MissingPivotDataset("Cannot construct an image from no datasets")
    if frame_acquisition_time is None:
      raise MissingPivotDataset("Cannot construct an image from no datasets")

    super().__init__(FramedImage(raw_image))
    self._pivot = first_dataset
    self._frame_durations_ms = frame_times_ms
    self._frame_acquisition_time = frame_acquisition_time
    self._pixel_volume = numpy.array([first_dataset.SliceThickness, first_dataset.PixelSpacing[1], first_dataset.PixelSpacing[0]])

  def __getattribute__(self, name: str) -> Any:
    try:
      return super().__getattribute__(name)
    except AttributeError:
      return self._pivot.__getattribute__(name)

__all__ = [
  'Series',
  'DicomSeries',
  'NiftiSeries',
]
