"""

  Requires numpy
"""
from enum import Enum
from functools import lru_cache, wraps
from inspect import getfullargspec

from pydicom import DataElement, Dataset
from pydicom.tag import BaseTag, Tag
from typing import Dict, List, Union, Tuple, Any, Optional, Callable, Iterator

from typing_extensions import TypedDict, NotRequired

from dicomnode.lib.dicom import make_meta, gen_uid
from dicomnode.lib.dicomFactory import AttrElement, CallElement, CopyElement, DicomFactory, Header, SeriesElement, FillingStrategy, StaticElement, HeaderBlueprint
from dicomnode.lib.exceptions import IncorrectlyConfigured, InvalidTagType, InvalidEncoding

import numpy
from numpy import ndarray

unsigned_array_encoding: Dict[int, type] = {
  8 : numpy.uint8,
  16 : numpy.uint16,
  32 : numpy.uint32,
  64 : numpy.uint64,
}

class NumpyCallerArgs(TypedDict):
  scaled_image: NotRequired[ndarray]
  slope : NotRequired[float]
  intercept : NotRequired[float]
  i : int
  image : ndarray # Unmodified image
  factory : 'NumpyFactory'
  VirtualElement : NotRequired['NumpyCaller'] # This is required However when this is created, it's missing

class NumpyCaller(CallElement):
  """Virtual Data Element, indicating that the value will produced from
  a provided callable function"""
  def __init__(self,
               tag: Union[int, str, Tuple[int, int]],
               VR: str,
               func: Callable[[],Any]
    ) -> None:
    self.tag: Union[int, str, Tuple[int, int]] = tag
    self.VR: str = VR
    self.func: Callable[[],Any] = func

  def __call__(self, **kwargs) -> DataElement:
    kwargs['VirtualElement'] = self
    value = self.func(**kwargs)
    if value is not None:
      return DataElement(self.tag, self.VR, value)
    else:
      return None


class NumpyFactory(DicomFactory):
  _bits_allocated: int = 16
  _bits_stored: int = 16
  _high_bit: int = 15
  _pixel_representation: int = 0

  def __init__(self,
               header_blueprint: Optional[HeaderBlueprint] = None,
               filling_strategy: Optional[FillingStrategy] = FillingStrategy.DISCARD) -> None:
    super().__init__(header_blueprint, filling_strategy)

  @property
  def pixel_representation(self) -> int:
    """Indicates encoding of pixel:
      0 : unsigned
      1 : two's compliment encoding
    """
    return self._pixel_representation

  @pixel_representation.setter
  def pixel_representation_setter(self, val):
    if not isinstance(val, int):
      raise TypeError("pixel representation must be an int")
    if val == 0 or val == 1:
      self._pixel_representation = val
    else:
      raise ValueError("Pixel Representation can take two values 0 or 1")

  @property
  def bits_allocated(self) -> int:
    """Determines how many bits will be allocated per pixel
    Must be 1 or a positive multiple of 8

    Defaults to 16
    """
    return self._bits_allocated

  @bits_allocated.setter
  def bits_allocated_setter(self, val: int) -> None:
    if not isinstance(val, int):
      raise TypeError("Bits allocated must be a positive integer of multiple 8")
    if val > 0 and (val == 1 or (val % 8 == 0)):
      self._bits_allocated = val
    else:
      raise ValueError("Bits allocated must be 1 or a positive multiple of 8")

  @property
  def bits_stored(self) -> int:
    """Defines how many of the bits will be used, must be less than allocated

    Defaults to 15
    """
    return self._bits_stored

  @bits_stored.setter
  def bits_stored_setter(self, val: int):
    if not isinstance(val, int):
      raise TypeError("bits stored must be an int")
    if 0 < val <= self.bits_allocated:
      self._bits_stored = val
    else:
      error_message = f"bits stored must be in range [1 - {self.bits_allocated}]"
      raise ValueError(error_message)

  @property
  def high_bit(self) -> int:
    """Defines the high bit, must be equal to bit stored - 1

    Defaults to 15
    """
    return self._high_bit

  @high_bit.setter
  def high_bit_setter(self, val: int):
    if not isinstance(val, int):
      raise TypeError("High bit must be an int")
    if val + 1 == self.bits_stored:
      self._high_bit = val
    else:
      error_message = f"high bit must equal to {self.bits_stored - 1}"
      raise ValueError(error_message)

  def scale_image(self, image: ndarray) -> Tuple[ndarray, float, float]:
    target_datatype = unsigned_array_encoding.get(self.bits_allocated, None)
    min_val = image.min()
    max_val = image.max()

    image_max_value = ((1 << self.bits_stored) - 1)

    slope = image_max_value / ( max_val - min_val)
    intercept = - min_val * slope

    new_image = (image * slope + intercept).astype(target_datatype)
    return new_image, slope, intercept


  def make_series(self, header : Header, image: ndarray):
    target_datatype = unsigned_array_encoding.get(self.bits_allocated, None)
    if target_datatype is None:
      raise IncorrectlyConfigured("There's no target Datatype")

    encode = image.dtype == target_datatype

    list_dicom = []
    if len(image.shape) == 3:
      for i, slice in enumerate(image):
        caller_args: NumpyCallerArgs = {
          'factory' : self,
          'i' : i,
          'image' : slice
        }

        # Encoding is done per slice basis
        if encode:
          scaled_slice, slope, intercept = self.scale_image(slice)
          caller_args['scaled_image'] = scaled_slice
          caller_args['slope'] = slope
          caller_args['intercept'] = intercept

        dataset = Dataset()
        for element in header:
          if isinstance(element, DataElement):
            dataset.add(element)
          if isinstance(element, CallElement):
            if data_element := element(**caller_args):
              dataset.add(data_element)
        list_dicom.append(dataset)
    return list_dicom

def _get_image(**kwargs: NumpyCallerArgs) -> ndarray:
  if 'scaled_image' in kwargs:
    image = kwargs['scaled_image']
  else:
    image = kwargs['image']
  return image

def _add_InstanceNumber(**kwargs: NumpyCallerArgs):
  return kwargs['i']

def _add_Rows(**kwargs: NumpyCallerArgs) -> int:
  image = _get_image(**kwargs)
  return image.shape[0]

def _add_Columns(**kwargs: NumpyCallerArgs) -> int:
  image = _get_image(**kwargs)
  return image.shape[1]

def _add_smallest_pixel(**kwargs: NumpyCallerArgs) -> int:
  image = _get_image(**kwargs)
  return image.min()

def _add_largest_pixel(**kwargs: NumpyCallerArgs) -> int:
  image = _get_image(**kwargs)
  return image.max()

def _add_aspect_ratio(**kwargs: NumpyCallerArgs) -> List[int]:
  image = _get_image(**kwargs)
  return [image.shape[0], image.shape[1]]

def _add_slope(**kwargs: NumpyCallerArgs) -> Optional[int]:
  return kwargs.get('slope')

def _add_intercept(**kwargs: NumpyCallerArgs) -> Optional[int]:
  return kwargs.get('intercept')

def _add_PixelData(**kwargs) -> bytes:
  image = _get_image(**kwargs)
  return image.tobytes()

####### Header Tag groups #######
general_image_header_tags = []

image_pixel_header_tags: HeaderBlueprint = HeaderBlueprint([
  StaticElement(0x00280002, 'US', 1),                    # SamplesPerPixel
  StaticElement(0x00280004, 'CS', 'MONOCHROME2'),        # PhotometricInterpretation
  NumpyCaller(0x00280010, 'US', _add_Rows),              # Rows
  NumpyCaller(0x00280011, 'US', _add_Columns),           # Columns
  NumpyCaller(0x00280034, 'IS', _add_aspect_ratio),      # PixelAspectRatio
  AttrElement(0x00280100, 'US', 'bits_allocated'),       # BitsAllocated
  AttrElement(0x00280101, 'US', 'bits_stored'),          # BitsStored
  AttrElement(0x00280102, 'US', 'high_bit'),             # HighBit
  AttrElement(0x00280103, 'US', 'pixel_representation'), # PixelRepresentation
  NumpyCaller(0x00280106, 'US', _add_smallest_pixel),    # SmallestImagePixelValue
  NumpyCaller(0x00280107, 'US', _add_largest_pixel),     # LargestImagePixelValue
  NumpyCaller(0x00281052, 'DS', _add_intercept),         # RescaleIntercept
  NumpyCaller(0x00281053, 'DS', _add_slope),             # RescaleSlope
  NumpyCaller(0x7FE00010, 'OB', _add_PixelData)          # PixelData
])


