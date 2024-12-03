"""This module contains the class series, which is an representation of a dicom
series. This class aims to provide a unified view of a series.

https://xkcd.com/927/
"""

# Python standard library
from datetime import datetime
from functools import reduce
from enum import Enum
from logging import getLogger
from typing import Any, Callable, Dict, List, Iterable, Literal, Optional,\
  Tuple, TypeAlias, Union

# Third party packages
import numpy
from numpy import ndarray, float32
from pydicom import Dataset, DataElement
from pydicom.tag import Tag
from pydicom.datadict import dictionary_VR, keyword_dict
from pydicom.tag import BaseTag
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from rt_utils import RTStruct

# Dicomnode packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.dicom import has_tags, sort_datasets, gen_uid
from dicomnode.math.space import Space, ReferenceSpace
from dicomnode.math.image import Image, FramedImage
from dicomnode.lib.exceptions import InvalidDataset, MissingPivotDataset, IncorrectlyConfigured

logger = getLogger(DICOMNODE_LOGGER_NAME)

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

  It's a lazy wrapper for image class, providing the image construction on use
  rather than on construction.
  """
  @property
  def image(self):
    if self._image is None:
      if isinstance(self._image_constructor, Callable): #pragma: no cover
        self._image = self._image_constructor()
      else:
        raise IncorrectlyConfigured("An Image must have an image or constructor")
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
  """This represent a series of dicom, that contains an image.

  Note that the image is constructed lazily, so that
  """
  pivot: Dataset
  datasets: List[Dataset]

  def __init__(self, datasets: List[Dataset]) -> None:
    if len(datasets) == 0:
      raise ValueError("Cannot construct a dicom series from an empty list")

    self.datasets = datasets
    self.pivot = self.datasets[0]
    if 'InstanceNumber' in self.pivot:
      self.datasets.sort(key=sort_datasets)
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
    """Returns the number of datasets in the series

    Returns:
        int : Returns the number of datasets in the series
    """
    # This Docs string is kinda pointless, or rather it doesn't show up in
    # various linting. :(
    return len(self.datasets)

  def __getitem__(self, tag) -> Optional[Union[DataElement, List[DataElement]]]:
    if tag in SERIES_VARYING_TAGS:
      return [dataset.get(tag, None) for dataset in self.datasets]
    return self.pivot.get(tag, None)

  def __getattribute__(self, name: str) -> Any:
    try:
      return super().__getattribute__(name)
    except AttributeError:
      tag = keyword_dict[name]

      return self.datasets[0][tag].value

  def __setitem__(self, tag: Union[int, str], value):
    if isinstance(tag, str):
      tag = Tag(tag)

    if tag in SERIES_VARYING_TAGS:
      if not isinstance(value, List):
        error_message = f"The tag is a varying dicom tag. The correct type is a list of length {len(self)}"
        raise TypeError(error_message)
      self.set_individual_tag(tag, value)
    else:
      self.set_shared_tag(tag, value)

  def set_shared_tag(self, tag: int, value: Any):
    for dataset in self.datasets:
      if not isinstance(value, DataElement):
        value = DataElement(tag, dictionary_VR(tag), value)
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
    return image.shape[0] == len(self.datasets)

  def shared_tag(self, tag) -> bool:
    return shared_tag(self.datasets, tag)

class NiftiSeries(Series):
  def __init__(self, nifti: Nifti1Image) -> None:
    self.nifti = nifti
    image_data = self.nifti.get_fdata()
    if image_data.flags.f_contiguous:
      image_data = numpy.transpose(image_data, [i for i in range(image_data.ndim)].reverse())
    affine = Space.from_nifti(self.nifti)

    super().__init__(Image(image_data, affine))

class FramedDicomSeries(Series):
  """A Series constructed from dicom datasets, which contains frames such as
  gates or dynamic series. Each frame is a 3 dimensional image of float32
  points which share a coordinate system.

  Each frame have identical Dimensions.

  Args:
    datasets (List[Dataset]): A non empty list of datasets where each dataset
      have the following tags:
      * NumberOfSlices
      * Rows
      * Columns
      * RescaleIntercept
      * RescaleSlope
      * PixelData
      * ActualFrameDuration
      * AcquisitionTime
      * AcquisitionDate
      * ImageIndex
      * PixelData
  """

  REQUIRED_TAGS = [
    'NumberOfSlices',
    'Rows',
    'Columns',
    'RescaleIntercept',
    'RescaleSlope',
    'PixelData',
    'ActualFrameDuration',
    'AcquisitionTime',
    'AcquisitionDate',
    'ImageIndex',
    'PixelData',
  ]

  class FRAME_TYPE(Enum):
    GATED = 1
    DYNAMIC = 2

  @property
  def image(self) -> FramedImage:
    image = super().image
    if not isinstance(image, FramedImage):
      raise IncorrectlyConfigured
    return image

  def frame(self, frame_number:int):
    return self.image.frame(frame_number)

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

    datasets_dict: Dict[int, List[Dataset]] = {}

    def add_dataset(dataset: Dataset):
      frame = dataset.ImageIndex // dataset.NumberOfSlices

      if frame in datasets_dict:
        datasets_dict[frame].append(dataset)
      else:
        datasets_dict[frame] = [dataset]


    def insert_image(raw:ndarray[Tuple[int,int,int,int], Any], dataset: Dataset):
      if not has_tags(dataset, self.REQUIRED_TAGS):
        missing_tags = []

        for tag in self.REQUIRED_TAGS:
          if tag not in dataset:
            missing_tags.append(str(tag))

        raise InvalidDataset(f"Dataset doesn't appear to be large pet as dataset is missing {' '.join(missing_tags)}")

      frame = dataset.ImageIndex // dataset.NumberOfSlices
      slice_number_in_series = dataset.ImageIndex % dataset.NumberOfSlices

      raw[frame, slice_number_in_series, :, :] = \
        dataset.pixel_array.astype(float32) * dataset.RescaleSlope\
          + dataset.RescaleIntercept

    for dataset in datasets:
      frames = dataset.NumberOfTimeSlices if 'NumberOfTimeSlices' in dataset else dataset.NumberOfTimeSlots
      if first_dataset is None:
        first_dataset = dataset

      if raw_image is None:
        raw_image = numpy.empty((frames, dataset.NumberOfSlices, dataset.Rows, dataset.Columns), float32)

      if frame_times_ms is None:
        frame_times_ms = numpy.ones((frames), dtype=numpy.int32)

      if frame_acquisition_time is None:
        frame_acquisition_time = numpy.ones((frames), dtype='datetime64[ms]')

      frameIndex = dataset.ImageIndex // dataset.NumberOfSlices
      insert_image(raw_image, dataset)
      add_dataset(dataset)
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

    space = Space.from_datasets(datasets_dict[0])

    super().__init__(FramedImage(raw_image, space))
    self.datasets = datasets_dict
    self._pivot = first_dataset
    self._frame_durations_ms = frame_times_ms
    self._frame_acquisition_time = frame_acquisition_time
    self._pixel_volume = numpy.array([first_dataset.SliceThickness, first_dataset.PixelSpacing[1], first_dataset.PixelSpacing[0]])

  def __getattribute__(self, name: str) -> Any:
    try:
      return super().__getattribute__(name)
    except AttributeError:
      return self._pivot.__getattribute__(name)

ImageContainerType = Union[
  Image,
  Nifti1Image,
  Nifti2Image,
  List[Dataset],
  Series,
  FramedImage
]

def extract_image(source: ImageContainerType, frame=None, mask=None) -> Image:
  if isinstance(source, Nifti1Image) or isinstance(source, Nifti2Image):
    source = NiftiSeries(source)
  if isinstance(source, List):
    source = DicomSeries(source)
  if isinstance(source, Series):
    source = source.image

  if isinstance(source, FramedImage):
    if frame is not None:
      return source.frame(frame)
    error_message = "Underlying Image is a framed image and not an plain image, use the frame to select frame"
    logger.error(error_message)
    raise TypeError(error_message)
  elif isinstance(source, Image):
    return source

  error_message = f"Unable to convert {type(source)} to an Image"
  logger.error(error_message)
  raise TypeError(error_message)

def extract_space(source: ImageContainerType) -> Space:
  if isinstance(source, Nifti1Image) or isinstance(source, Nifti2Image):
    source = NiftiSeries(source)
  if isinstance(source, List):
    source = DicomSeries(source)

  if isinstance(source, Series):
    return source.image.space

  if isinstance(source, Image):
    return source.space

  if isinstance(source, FramedImage):
    return source.space

  raise TypeError(f"Cannot extract space from object of type {type(source)}")


def frame_unrelated_series(*frame_series: DicomSeries,
                           frame_type = FramedDicomSeries.FRAME_TYPE.DYNAMIC) -> FramedDicomSeries:
  """If you have multiple series, that you wish to unite in a single series,
  recreates SOPInstanceUID

  Args:
    *

  Raises:
      ValueError: _description_
  """
  number_of_frames = len(frame_series)
  number_of_datasets = None
  series_uid = gen_uid()
  index = 0

  datasets: List[Dataset] = []

  if number_of_frames == 0:
    raise ValueError("Cannot Relate zero series")

  for i, series in enumerate(frame_series):
    if number_of_datasets is None:
      number_of_datasets = len(series)
    elif number_of_datasets != len(series):
      raise ValueError(f"Series {i + 1} doesn't contain {number_of_datasets} which the other datasets do!")

    indexes = [i + 1 for i in range(index, index + number_of_datasets)]
    series["SOPInstanceUID"] = [gen_uid() for _ in series]
    series["SeriesInstanceUID"] = series_uid
    series["InstanceNumber"] = indexes
    series["ImageIndex"] = indexes
    series["NumberOfSlices"] = number_of_datasets
    if frame_type == FramedDicomSeries.FRAME_TYPE.DYNAMIC:
      series["NumberOfTimeSlices"] = number_of_frames
    elif frame_type == FramedDicomSeries.FRAME_TYPE.GATED:
      series['NumberOfRRIntervals'] = 1
      series['NumberOfTimeSlots'] = number_of_frames

    datasets.extend(series.datasets)

  return FramedDicomSeries(datasets)

__all__ = [
  'Series',
  'DicomSeries',
  'NiftiSeries',
]
