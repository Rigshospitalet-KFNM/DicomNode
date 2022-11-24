"""

  Requires numpy
"""

from enum import Enum

from inspect import getfullargspec

from pydicom import DataElement, Dataset
from pydicom.tag import BaseTag, Tag
from typing import Dict, List, Union, Tuple, Any, Optional, Callable, Iterator

from dicomnode.lib.dicom import make_meta, gen_uid
from dicomnode.lib.dicomFactory import AttrElement, CallElement,CopyElement, DicomFactory, Header, SeriesElement, FillingStrategy, StaticElement, HeaderBlueprint
from dicomnode.lib.exceptions import IncorrectlyConfigured, InvalidTagType

import numpy
from numpy import ndarray

class NumpyCaller(CallElement):
  """Virtual Data Element, indicating that the value will produced from
  a provided callable function"""
  def __init__(self,
               tag: Union[int, str, Tuple[int, int]],
               VR: str,
               func: Union[Callable[[],Any],
                           Callable[['ndarray'], Any],
                           Callable[[int, 'ndarray'], Any]]
    ) -> None:
    self.tag: Union[int, str, Tuple[int, int]] = tag
    self.VR: str = VR
    self.func: Union[Callable[[],Any],
                     Callable[[ndarray],Any],
                     Callable[[int, ndarray], Any]] = func
    self.func_require_image: bool = len(getfullargspec(func).args) == 1
    self.func_require_all: bool = len(getfullargspec(func).args) == 2

  def __call__(self, i : int, slice: ndarray) -> DataElement:
    if self.func_require_all:
      value = self.func(i, slice)
    elif self.func_require_image:
      value = self.func(slice)
    else:
      value = self.func()
    return DataElement(self.tag, self.VR, value)



class NumpyFactory(DicomFactory):
  def __init__(self,
               header_blueprint: Optional[List[Union[DataElement, CopyElement, CallElement]]] = None,
               filling_strategy: Optional[FillingStrategy] = FillingStrategy.DISCARD) -> None:
    super().__init__(header_blueprint, filling_strategy)
    self.bits_allocated = 16
    self.bits_stored = 16
    self.high_bit = 15

  def make_series(self, header : Header, image: ndarray):
    list_dicom = []
    if len(image.shape) == 3:
      for i, slice in enumerate(image):
        slice: ndarray = slice # Just here for
        dataset = Dataset()
        for element in header:
          if isinstance(element, DataElement):
            dataset.add(element)
          if isinstance(element, CallElement):
            dataset.add(element(i, slice))
        list_dicom.append(dataset)
    return list_dicom



####### Call element callables #######

def _add_InstanceNumber(i: int, _: ndarray):
  return i

def _add_Rows(image: ndarray) -> int:
  return image.shape[0]

def _add_Columns(image: ndarray) -> int:
  return image.shape[1]

def _add_aspect_ratio(image: ndarray) -> List[int]:
  return [image.shape[0], image.shape[1]]

def _add_PixelData(image: ndarray) -> bytes:
  return image.tobytes()

####### Header Tag groups #######

image_pixel_header_tags: HeaderBlueprint = HeaderBlueprint([
  DataElement(0x00280002, 'US', 1),                 # SamplesPerPixel
  DataElement(0x00280004, 'CS', 'MONOCHROME2'),     # PhotometricInterpretation
  NumpyCaller(0x00280010, 'US', _add_Rows),         # Rows
  NumpyCaller(0x00280010, 'US', _add_Columns),      # Columns
  NumpyCaller(0x00280034, 'IS', _add_aspect_ratio), # PixelAspectRatio
  AttrElement(0x00280100, 'US', 'bits_allocated'),  # BitsAllocated
  AttrElement(0x00280101, 'US', 'bits_stored'),     # BitsStored
  AttrElement(0x00280102, 'US', 'high_bit'),        # HighBit
  DataElement(0x00280103, 'US', 0),                 # PixelRepresentation
  NumpyCaller(0x7FE00010, 'OB', _add_PixelData)     # PixelData
])



