"""

  Requires numpy
"""
from pydicom import DataElement, Dataset
from pydicom.tag import BaseTag, Tag
from typing import Dict, List, Union, Tuple, Any, Optional, Callable, Iterator

from dataclasses import dataclass

from dicomnode.lib.dicom import make_meta, gen_uid
from dicomnode.lib.dicomFactory import AttrElement, CallerArgs, CallElement, DicomFactory, SeriesHeader,\
  StaticElement, Blueprint, patient_blueprint, general_series_blueprint,\
  general_study_blueprint, SOP_common_blueprint, frame_of_reference_blueprint,\
  general_equipment_blueprint, general_image_blueprint, ct_image_blueprint
from dicomnode.lib.exceptions import IncorrectlyConfigured, InvalidTagType, InvalidEncoding

import numpy
from numpy import ndarray

unsigned_array_encoding: Dict[int, type] = {
  8 : numpy.uint8,
  16 : numpy.uint16,
  32 : numpy.uint32,
  64 : numpy.uint64,
}

@dataclass
class NumpyCallerArgs(CallerArgs):
  image : ndarray # Unmodified image
  factory : 'NumpyFactory'
  intercept : Optional[float] = None
  slope : Optional[float] = None
  scaled_image: Optional[ndarray] = None
  total_images: Optional[int] = None

class NumpyCaller(CallElement):
  """Virtual Data Element, indicating that the value will produced from
  a provided callable function"""
  def __init__(self,
               tag: Union[int, str, Tuple[int, int]],
               VR: str,
               func: Callable[[NumpyCallerArgs],Any]
    ) -> None:
    self.tag: Union[int, str, Tuple[int, int]] = tag
    self.VR: str = VR
    self.func: Callable[[NumpyCallerArgs],Any] = func

  def __call__(self, caller_args : NumpyCallerArgs) -> Optional[DataElement]:
    """Now There's a Liskov's Substitution principle violation here,
    because NumpyCaller is more inherits from Caller Args.

    Args:
        caller_args (NumpyCallerArgs): _description_

    Returns:
        Optional[DataElement]: _description_
    """
    caller_args.virtual_element = self
    value = self.func(caller_args)
    if value is not None:
      return DataElement(self.tag, self.VR, value)
    else:
      return None


class NumpyFactory(DicomFactory):
  _bits_allocated: int = 16
  _bits_stored: int = 16
  _high_bit: int = 15
  _pixel_representation: int = 0

  @property
  def pixel_representation(self) -> int:
    """Indicates encoding of pixel:
      0 : unsigned
      1 : two's compliment encoding
    """
    return self._pixel_representation

  @pixel_representation.setter
  def pixel_representation(self, val):
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
  def bits_allocated(self, val: int) -> None:
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
  def bits_stored(self, val: int):
    if not isinstance(val, int):
      raise TypeError("bits stored must be an int") # type: ignore
    if 1 <= val <= self.bits_allocated:
      self._bits_stored = val
    else:
      raise ValueError(f"bit stored must be in range [1, {self.bits_allocated}]")

  @property
  def high_bit(self) -> int:
    """Defines the high bit, must be equal to bit stored - 1

    Defaults to 15
    """
    return self._high_bit

  @high_bit.setter
  def high_bit(self, val: int):
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


  def build_from_header(self, header : SeriesHeader, image: ndarray) -> List[Dataset]:
    target_datatype = unsigned_array_encoding.get(self.bits_allocated, None)
    if target_datatype is None:
      raise IncorrectlyConfigured("There's no target Datatype") # pragma: no cover this might happen, if people are stupid

    encode = image.dtype == target_datatype

    list_dicom = []
    if len(image.shape) == 3:
      for i, slice in enumerate(image):
        caller_args = NumpyCallerArgs(
          i=i,
          factory=self,
          image=slice
        )

        caller_args.total_images = image.shape[2]

        # Encoding is done per slice basis
        if encode:
          scaled_slice, slope, intercept = self.scale_image(slice)
          caller_args.scaled_image = scaled_slice
          caller_args.slope = slope
          caller_args.intercept = intercept

        dataset = Dataset()
        for element in header:
          if isinstance(element, DataElement):
            dataset.add(element)
          elif isinstance(element, CallElement):
            data_element = element(caller_args)
            if data_element is not None:
              dataset.add(data_element)
        list_dicom.append(dataset)
    return list_dicom

def _get_image(numpy_caller_args: NumpyCallerArgs) -> ndarray:
  if numpy_caller_args.scaled_image is not None:
    image = numpy_caller_args.scaled_image
  else:
    image = numpy_caller_args.image
  return image


def _add_Rows(numpy_caller_args: NumpyCallerArgs) -> int:
  image = _get_image(numpy_caller_args)
  return image.shape[0]

def _add_Columns(numpy_caller_args: NumpyCallerArgs) -> int:
  image = _get_image(numpy_caller_args)
  return image.shape[1]

def _add_smallest_pixel(numpy_caller_args: NumpyCallerArgs) -> int:
  image = _get_image(numpy_caller_args)
  return image.min()

def _add_largest_pixel(numpy_caller_args: NumpyCallerArgs) -> int:
  image = _get_image(numpy_caller_args)
  return image.max()

def _add_aspect_ratio(numpy_caller_args: NumpyCallerArgs) -> List[int]:
  image = _get_image(numpy_caller_args)
  return [image.shape[0], image.shape[1]]

def _add_images_in_acquisition(numpy_caller_args: NumpyCallerArgs) -> Optional[int]:
  return numpy_caller_args.total_images

def _add_slope(numpy_caller_args: NumpyCallerArgs) -> Optional[float]:
  return numpy_caller_args.slope

def _add_intercept(numpy_caller_args: NumpyCallerArgs) -> Optional[float]:
  return numpy_caller_args.intercept

def _add_PixelData(numpy_caller_args: NumpyCallerArgs) -> bytes:
  image = _get_image(numpy_caller_args)
  return image.tobytes()

####### Header Tag groups #######
image_pixel_NumpyBlueprint: Blueprint = Blueprint([
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
  NumpyCaller(0x7FE00010, 'OB', _add_PixelData),          # PixelData,
  # There's some General Image tags in here because, they don't know total images in that definition
  NumpyCaller(0x00201002, 'IS', _add_images_in_acquisition)
])

CTImageStorage_NumpyBlueprint: Blueprint = patient_blueprint\
                                           + general_study_blueprint \
                                           + general_series_blueprint \
                                           + frame_of_reference_blueprint \
                                           + general_equipment_blueprint \
                                           + general_image_blueprint \
                                           + image_pixel_NumpyBlueprint\
                                           + ct_image_blueprint \
                                           + SOP_common_blueprint

