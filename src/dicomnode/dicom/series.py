"""This module contains the class series, which is an representation of a dicom
series. This class aims to provide a unified view of a series.

https://xkcd.com/927/
"""

# Python standard library
from datetime import datetime
from functools import reduce
from enum import Enum
from operator import mul, and_
from logging import getLogger
from typing import Any, Callable, Dict, List, Iterable, Literal, Optional,\
  Tuple, TypeAlias, Union

# Third party packages
import numpy
from numpy import ndarray, float32
from pydicom import Dataset, DataElement
from pydicom.uid import UID
from pydicom.tag import Tag
from pydicom.datadict import dictionary_VR, keyword_dict
from pydicom.tag import BaseTag
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from rt_utils import RTStruct

# Dicomnode packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.dicom import has_tags, sort_datasets, gen_uid,\
  assess_single_series
from dicomnode.math import transpose_nifti_coords
from dicomnode.math.space import Space, ReferenceSpace
from dicomnode.math.image import Image
from dicomnode.lib.exceptions import MissingDatasets, InvalidDataset, MissingPivotDataset, IncorrectlyConfigured

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

  pivot = datasets[0]
  return reduce(and_,
                [dataset.get(tag, None) == pivot.get(tag, None)
                  for dataset in datasets],
                True)

SERIES_VARYING_TAGS = set([
  0x0008_0018, # SOPInstanceUID
  0x0020_0013, # InstanceNumber
  0x0020_0032, # ImagePositionPatient
  0x0020_1041, # SliceLocation
  0x0028_0106, # SmallestImagePixelValue
  0x0028_0107, # LargestImagePixelValue
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
  def image(self) -> Image:
    if self._image is None:
      if isinstance(self._image_constructor, Callable): #pragma: no cover
        self._image = self._image_constructor()
      else:
        raise IncorrectlyConfigured("An Image must have an image or constructor")
    return self._image

  @image.setter
  def image_setter(self, new_image: Image | Callable[[], Image]):
    if isinstance(new_image, Image):
      self._image = new_image
    elif isinstance(new_image, Callable):
      self._image = None
      self._image_constructor = new_image


  # Constructors
  def __init__(self, image: Union[Image, Callable[[],Image]]):
    if isinstance(image, Image):
      self._image = image
      self._image_constructor = None
    else:
      self._image = None
      self._image_constructor = image

  def to_nifti(self):
    return Nifti1Image(
      transpose_nifti_coords(self.image.raw),
      self.image.space.to_affine()
    )

  def slices(self):
    """Yield each image slice in the underlying image

    Yields:
        NDArray[float]: An 2D image slice in the image
    """
    yield from self.image.slices()


class DicomSeries(Series):
  """This represent a series of dicom datasets, that together contains an image.

  Note: that the image is constructed lazily, so that
  """
  pivot: Dataset
  datasets: List[Dataset]
  series_instance_UID: Optional[UID]

  def _image_constructor(self):
      return Image.from_datasets(self.datasets)

  def __init__(self, datasets: List[Dataset]) -> None:
    if len(datasets) == 0:
      raise MissingDatasets("Cannot construct a dicom series from an empty list")

    self.series_instance_UID = assess_single_series(datasets)

    self.datasets = datasets
    self.pivot = self.datasets[0]
    if 'InstanceNumber' in self.pivot:
      self.datasets.sort(key=sort_datasets)
      self.pivot = self.datasets[0]
    else:
      self.set_individual_tag(0x0020_0013, [i + 1 for i,_ in enumerate(self.datasets)])

    super().__init__(self._image_constructor)

  @classmethod
  def create_skeleton(cls, image: Image | ndarray):
    """Alternative constructor.
    Creates a series with empty datasets such that each dataset could take a 2D
    Slice. Works for any dimensional greater than 2.

    Note that if you access the image attributes, you'll encounter an error,
    until you have filled the datasets.

    Args:
        image (Image): The Image, which you intent to create an image for

    Returns:
        DicomSeries: An empty series with only InstanceNumber Attribute
    """
    if not isinstance(image, Image):
      image = Image.from_array(image)

    datasets = []

    for i, slice_ in enumerate(image.slices()):
      dataset = Dataset()
      dataset.InstanceNumber = i + 1

      datasets.append(dataset)
    return cls(datasets)

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
    # This is here to ensure that a dicom series is picklable
    if name.startswith('_'):
      return super().__getattribute__(name)

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
    """Sets a tag in the series with an individual value for each datasets.

    Args:
        tag (int): The Tag, that you wish to set
        values (List[Union[DataElement, Any]]): A list of values that you wish to set

    Raises:
        ValueError: Raised when the length of values is not equal to the amount
                    of datasets in the series
    """

    if len(values) != len(self):
      error_message = f"The amount of values ({len(values)}) doesn't match the amount datasets ({len(self)})"
      raise ValueError(error_message)
    for dataset, value in zip(self.datasets, values):
      if not isinstance(value, DataElement):
        value = DataElement(tag, dictionary_VR(tag), value)
      dataset[tag] = value

  def set_frame_tags(self, tags: List[DataElement]):
    """Sets a tag that is individual to each frame.

    Args:
        tags (List[DataElement]): A list of DataElements, that will be set

    Raises:
      Invalid Dataset: Raised when the series is not at least four dimensional
      ValueError: Raised when tags are not of length equal to number of frames
    """
    if self.image.raw.ndim >= 4:
      raise InvalidDataset("You cannot get frame values from a 3 dimensional image")

    frames = self.image.raw.shape[-4]
    if len(tags) != frames:
       raise ValueError(f"The number of frames {frames} do not match the length values: {len(tags)}")

    images_per_frame = self.image.raw.shape[-3]

    for i, dataset in enumerate(self):
      frame = i // images_per_frame
      tag = tags[frame]
      dataset[tag.tag] = tag

  def get_frame_values(self, tag: int):
    if self.image.raw.ndim >= 4:
      raise InvalidDataset("You cannot get frame values from a 3 dimensional image")

    images_per_frame = self.image.raw.shape[-3]
    values = []

    for frame_number in range(self.image.raw[-4]):
      dataset = self.datasets[frame_number * images_per_frame]
      values.append(dataset[tag].value)

    return values


  def can_copy_into_image(self, image:ndarray[Tuple[int,int,int],Any]) -> bool:
    return image.shape[-3] == len(self.datasets)

  def shared_tag(self, tag) -> bool:
    return shared_tag(self.datasets, tag)

  def frame(self, frame: int) -> Image:
    return self.image.frame(frame)

class NiftiSeries(Series):
  def __init__(self, nifti: Nifti1Image) -> None:
    self.nifti = nifti
    image_data = self.nifti.get_fdata()
    if image_data.flags.f_contiguous:
      #image_data = numpy.transpose(image_data, [i for i in range(image_data.ndim)].reverse())
      image_data = transpose_nifti_coords(image_data)
    affine = Space.from_nifti(self.nifti)

    super().__init__(Image(image_data, affine))

ImageContainerType = Union[
  Image,
  Nifti1Image,
  Nifti2Image,
  List[Dataset],
  Series,
]

def extract_image(source: ImageContainerType) -> Image:
  """Extracts an image from a containing class

  Args:
      source (ImageContainerType): _description_

  Raises:
      TypeError: _description_

  Returns:
      Image: _description_
  """
  if isinstance(source, Nifti1Image) or isinstance(source, Nifti2Image):
    source = NiftiSeries(source)
  if isinstance(source, List):
    source = DicomSeries(source)
  if isinstance(source, Series):
    source = source.image

  if isinstance(source, Image):
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

  raise TypeError(f"Cannot extract space from object of type {type(source)}")


def frame_unrelated_series(*frame_series: DicomSeries) -> DicomSeries:
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
    index += number_of_datasets
    series["SOPInstanceUID"] = [gen_uid() for _ in series]
    series["SeriesInstanceUID"] = series_uid
    series["InstanceNumber"] = indexes
    series["ImageIndex"] = indexes
    series["NumberOfSlices"] = number_of_datasets

    datasets.extend(series.datasets)

  return DicomSeries(datasets)
__all__ = [
  'Series',
  'DicomSeries',
  'NiftiSeries',
]
