"""Specialized DicomFactory for building dicom series from
numpy arrays

"""

# Python Standard Library
from typing import Dict, List, Union, Tuple, Any, Optional, Callable, Iterator

# Third party packages
import numpy
from numpy import ndarray
from pydicom import DataElement, Dataset
from pydicom.tag import BaseTag, Tag

# Dicomnode packages
from dicomnode.lib.logging import get_logger
from dicomnode.lib.dicom import make_meta, gen_uid
from dicomnode.lib.dicom_factory import AttributeElement, InstanceEnvironment, FunctionalElement, DicomFactory, SeriesHeader,\
  StaticElement, Blueprint, patient_blueprint, general_series_blueprint, \
  general_study_blueprint, SOP_common_blueprint, frame_of_reference_blueprint, \
  general_equipment_blueprint, general_image_blueprint, ct_image_blueprint, \
  image_plane_blueprint, InstanceVirtualElement
from dicomnode.lib.exceptions import IncorrectlyConfigured, InvalidTagType, InvalidEncoding


logger = get_logger()

class NumpyFactory(DicomFactory):
  _bits_allocated: int = 16
  _bits_stored: int = 16
  _high_bit: int = 15
  _pixel_representation: int = 0

  _unsigned_array_encoding: Dict[int, type] = {
    8 : numpy.uint8,
    16 : numpy.uint16,
    32 : numpy.uint32,
    64 : numpy.uint64,
  }

  _signed_array_encoding: Dict[int, type] = {
    8 : numpy.int8,
    16 : numpy.int16,
    32 : numpy.int32,
    64 : numpy.int64,
  }

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
    target_datatype = self._unsigned_array_encoding.get(self.bits_allocated, None)
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
      return image.astype(target_datatype), 1, 0

    image_max_value = ((1 << self.bits_stored) - 1)

    slope = (max_val - min_val) / image_max_value
    intercept = min_val

    new_image = ((image - intercept) / slope).astype(target_datatype)

    return new_image, slope, intercept


  def build_from_header(self, header : SeriesHeader, image: ndarray) -> List[Dataset]:
    """This construct a dicom series from a header and numpy array containing
    the Image

    Args:
        header (SeriesHeader): _description_
        image (ndarray): _description_

    Raises:
        IncorrectlyConfigured: _description_

    Returns:
        List[Dataset]: _description_
    """
    target_datatype = self._unsigned_array_encoding.get(self.bits_allocated, None)
    if target_datatype is None:
      raise IncorrectlyConfigured("There's no target Datatype") # pragma: no cover this might happen, if people are stupid

    encode = image.dtype != target_datatype
    list_dicom = []
    if len(image.shape) == 3:
      logger.debug(f"Building dicom series of images {image.shape[0]} of dimension: {image.shape[2]}x{image.shape[1]} ")
      for i, slice in enumerate(image):
        instance_environment = InstanceEnvironment(
          instance_number= i + 1,
          factory=self,
          image=slice,
          total_images = image.shape[0],
        )

        # Encoding is done per slice basis
        if encode:
          scaled_slice, slope, intercept = self.scale_image(slice)
          instance_environment.scaled_image = scaled_slice
          instance_environment.slope = slope
          instance_environment.intercept = intercept

        dataset = Dataset()
        for element in header:
          if isinstance(element, DataElement):
            dataset.add(element)
          elif isinstance(element, InstanceVirtualElement):
            data_element = element.produce(instance_environment)
            if data_element is not None:
              dataset.add(data_element)
        make_meta(dataset)
        list_dicom.append(dataset)
    else:
      raise IncorrectlyConfigured("3 dimensional images are only supported") # pragma: no cover
    return list_dicom

def _get_image(instance_environment: InstanceEnvironment) -> ndarray:
  if instance_environment.scaled_image is not None:
    image = instance_environment.scaled_image
  else:
    image = instance_environment.image
  if image is None:
    raise IncorrectlyConfigured # pragma: no cover
  return image


def _add_Rows(instance_environment: InstanceEnvironment) -> int:
  image = _get_image(instance_environment)
  return image.shape[0]

def _add_Columns(instance_environment: InstanceEnvironment) -> int:
  image = _get_image(instance_environment)
  return image.shape[1]

def _add_smallest_pixel(instance_environment: InstanceEnvironment) -> int:
  image = _get_image(instance_environment)
  return int(image.min())

def _add_largest_pixel(instance_environment: InstanceEnvironment) -> int:
  image = _get_image(instance_environment)
  return int(image.max())

def _add_aspect_ratio(instance_environment: InstanceEnvironment) -> List[int]:
  image = _get_image(instance_environment)
  return [image.shape[0], image.shape[1]]

def _add_images_in_acquisition(instance_environment: InstanceEnvironment) -> Optional[int]:
  return instance_environment.total_images

def _add_slope(instance_environment: InstanceEnvironment) -> Optional[float]:
  return instance_environment.slope

def _add_intercept(instance_environment: InstanceEnvironment) -> Optional[float]:
  return instance_environment.intercept

def _add_PixelData(instance_environment: InstanceEnvironment) -> bytes:
  image = _get_image(instance_environment)
  return image.tobytes()

####### Header Tag groups #######
image_pixel_blueprint: Blueprint = Blueprint([
  StaticElement(0x00280002, 'US', 1),                    # SamplesPerPixel
  StaticElement(0x00280004, 'CS', 'MONOCHROME2'),        # PhotometricInterpretation
  FunctionalElement(0x00280010, 'US', _add_Rows),              # Rows
  FunctionalElement(0x00280011, 'US', _add_Columns),           # Columns
  FunctionalElement(0x00280034, 'IS', _add_aspect_ratio),      # PixelAspectRatio
  AttributeElement(0x00280100, 'US', 'bits_allocated'),       # BitsAllocated
  AttributeElement(0x00280101, 'US', 'bits_stored'),          # BitsStored
  AttributeElement(0x00280102, 'US', 'high_bit'),             # HighBit
  AttributeElement(0x00280103, 'US', 'pixel_representation'), # PixelRepresentation
  FunctionalElement(0x00280106, 'US', _add_smallest_pixel),    # SmallestImagePixelValue
  FunctionalElement(0x00280107, 'US', _add_largest_pixel),     # LargestImagePixelValue
  FunctionalElement(0x00281052, 'DS', _add_intercept),         # RescaleIntercept
  FunctionalElement(0x00281053, 'DS', _add_slope),             # RescaleSlope
  FunctionalElement(0x7FE00010, 'OB', _add_PixelData),          # PixelData,
  # There's some General Image tags in here because, they don't know total images in that definition
  FunctionalElement(0x00201002, 'IS', _add_images_in_acquisition),
])

CTImageStorage_NumpyBlueprint: Blueprint = patient_blueprint\
                                           + general_study_blueprint \
                                           + general_series_blueprint \
                                           + image_plane_blueprint \
                                           + frame_of_reference_blueprint \
                                           + general_equipment_blueprint \
                                           + general_image_blueprint \
                                           + image_pixel_blueprint\
                                           + ct_image_blueprint \
                                           + SOP_common_blueprint

