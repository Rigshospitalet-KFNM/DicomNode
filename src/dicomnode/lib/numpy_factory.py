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
from dicomnode.lib.dicom_factory import InstanceEnvironment, FunctionalElement, DicomFactory, SeriesHeader,\
  StaticElement, Blueprint, patient_blueprint, general_series_blueprint, \
  general_study_blueprint, SOP_common_blueprint, frame_of_reference_blueprint, \
  general_equipment_blueprint, general_image_blueprint, ct_image_blueprint, \
  image_plane_blueprint, InstanceVirtualElement
from dicomnode.lib.exceptions import IncorrectlyConfigured, InvalidTagType, InvalidEncoding


logger = get_logger()

class NumpyFactory(DicomFactory):

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

  def scale_image(self,
                  image: ndarray,
                  bits_stored = 16,
                  bits_allocated = 16,
                ) -> Tuple[ndarray, float, float]:
    target_datatype = self._unsigned_array_encoding.get(bits_allocated, None)
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
      return numpy.zeros_like(image), 1, min_val

    image_max_value = ((1 << bits_stored) - 1)

    slope = (max_val - min_val) / image_max_value
    intercept = min_val

    new_image = ((image - intercept) / slope).astype(target_datatype)

    return new_image, slope, intercept


  def build_from_header(self,
                        header : SeriesHeader,
                        image: ndarray,
                        kwargs: Dict[Any, Any] = {}) -> List[Dataset]:
    """This construct a dicom series from a header and numpy array containing
    the Image

    Args:
        header (SeriesHeader): The header which contains data
        image (ndarray): The image that will be produced into a dicom series
        kwargs (Dict[Any, Any]): Keyword arguments that is parsed to InstancedVirtualElement

    Raises:
        IncorrectlyConfigured: _description_

    Returns:
        List[Dataset]: _description_
    """
    bits_allocated_tag = header[0x00280100] # Bits Allocated
    if isinstance(bits_allocated_tag, DataElement):
      bits_allocated = bits_allocated_tag.value
    else:
      logger.error(f"Trying to build a Dicom series, but bit allocated is not in the Series header")
      raise Exception

    target_datatype = self._unsigned_array_encoding.get(bits_allocated, None)
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
          kwargs=kwargs
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
  StaticElement(0x00280100, 'US', 16),                         # BitsAllocated
  StaticElement(0x00280101, 'US', 16),                         # BitsStored
  StaticElement(0x00280102, 'US', 15),                         # HighBit
  StaticElement(0x00280103, 'US', 0), # PixelRepresentation
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

