""""""

__author__ = "Christoffer Vilstrup Jensen"


from abc import ABC, abstractmethod
from datetime import date, datetime, time
from enum import Enum
from inspect import getfullargspec
from pydicom import DataElement, Dataset
from pydicom.tag import Tag, BaseTag
from random import randint
from typing import Any, Callable, Dict, List, Iterator,  Optional, Tuple, Union

from dicomnode.lib.dicom import gen_uid
from dicomnode.lib.exceptions import InvalidTagType, IncorrectlyConfigured

class FillingStrategy(Enum):
  DISCARD = 0
  COPY = 1

class VirtualElement(ABC):
  """"""
  __tag: BaseTag

  @property
  def tag(self) -> BaseTag:
    """This is tag, that when the corporealized data element will have.

    Returns:
        int: int between 0 and 2 ** 32 - 1
    """
    return self.__tag

  @tag.setter
  def tag(self, tag: Union[BaseTag, str, int, Tuple[int,int]]):
    self.__tag = Tag(tag)

  @abstractmethod
  def corporealialize(self, factory: 'DicomFactory', dataset: Dataset) -> Optional[Union[DataElement, 'CallElement']]:
    raise NotImplemented # pragma: no cover


class AttrElement(VirtualElement):
  """Reads an attribute from the factory and creates a data element
  from it"""
  def __init__(self, tag: Union[BaseTag, str, int, Tuple[int,int]], VR: str, attribute: str) -> None:
    self.tag = tag
    self.VR = VR
    self.attribute = attribute

  def corporealialize(self, factory: 'DicomFactory', _: Dataset) -> DataElement:
    value = getattr(factory, self.attribute)
    return DataElement(self.tag, self.VR, value)

class CallElement(VirtualElement):
  """Abstract Virtual tag. This class represents a tag, that will be
  instantiated from with an image slice.
  """
  def __init__(self, tag, VR, func) -> None:
    self.tag: BaseTag = tag
    self.VR: str = VR
    self.func = func

  def corporealialize(self, _DF: 'DicomFactory', _DS: Dataset) -> 'CallElement':
    return self

  @abstractmethod
  def __call__(self) -> DataElement:
    return DataElement(self.tag, self.VR, self.func())

class CopyElement(VirtualElement):
  """Virtual Data Element, indicating that the value will be copied from an
  original dataset, Throws an error is element is missing"""

  def __init__(self, tag: Union[BaseTag, str, int, Tuple[int,int]], Optional: bool = False) -> None:
    self.tag: BaseTag = Tag(tag)
    self.Optional = Optional

  def corporealialize(self, _: 'DicomFactory', dataset: Dataset) -> Optional[DataElement]:
    if self.tag in dataset:
      return dataset[self.tag]
    else:
      if self.Optional:
        return None
      else:
        raise KeyError(f"{self.tag} not found in Header Parent Dataset")

class DiscardElement(VirtualElement):
  def __init__(self, tag) -> None:
    self.tag = tag

  def corporealialize(self, factory: 'DicomFactory', dataset: Dataset) -> Optional[Union[DataElement, 'CallElement']]:
    return None

class SeriesElement(VirtualElement):
  """This virtual element is instantiated when the header is created
  It may depend on input dataset"""
  def __init__(self,
               tag: Union[BaseTag, str, int, Tuple[int,int]],
               VR: str,
               func: Union[Callable[[Dataset], Any], Callable[[],Any]]) -> None:
    self.tag = tag
    self.VR  = VR
    self.func:  Union[Callable[[Dataset], Any], Callable[[],Any]] = func
    self.require_dataset: bool = len(getfullargspec(func).args) == 1

  def corporealialize(self, _: 'DicomFactory', dataset: Dataset) -> DataElement:
    if self.require_dataset:
      # The '# type: ignores' here is because the type checker assumes func
      # is an instance based function, when it is dynamic assigned function
      # and therefore behaves like a static function  
      value = self.func(dataset) # type: ignore 
    else:
      value = self.func() # type: ignore
    return DataElement(self.tag, self.VR, value)

class StaticElement(VirtualElement):
  def __init__(self, tag, VR, value) -> None:
    self.tag = tag
    self.VR = VR
    self.value = value

  def corporealialize(self, factory: 'DicomFactory', dataset: Dataset) -> Optional[Union[DataElement, 'CallElement']]:
    return DataElement(self.tag, self.VR, self.value)

class HeaderBlueprint():
  def __getitem__(self, __tag: int):
    return self._dict[__tag]

  def __setitem__(self, __tag: int, __ve: VirtualElement) -> None:
    if not isinstance(__tag, BaseTag):
      __tag = Tag(__tag)
    if not isinstance(__ve, VirtualElement):
      raise TypeError("HeaderBlueprint only accepts VirtualElements as element")
    if __tag != __ve.tag:
      raise ValueError("Miss match between tag and VirtualElement's tag")
    self.add_virtual_element(__ve)

  def __delitem__(self, tag):
    del self._dict[tag]

  def __init__(self, virtual_elements: List[VirtualElement] = []) -> None:
    # Init
    self._dict: Dict[int, VirtualElement] = {}

    # Fill from init
    for ve in virtual_elements:
      self.add_virtual_element(ve)

  def __len__(self) -> int:
    return len(self._dict)

  def __contains__(self, tag: int) -> bool:
    return tag in self._dict

  def __add__(self, blueprint: 'HeaderBlueprint') -> 'HeaderBlueprint':
    new_header_blueprint = type(self)()

    for ve in self:
      new_header_blueprint.add_virtual_element(ve)

    for ve in blueprint:
      new_header_blueprint.add_virtual_element(ve)

    return new_header_blueprint

  def __iter__(self) -> Iterator[VirtualElement]:
    for ve in self._dict.values():
      yield ve

  def add_virtual_element(self, virtual_element: VirtualElement):
    self._dict[virtual_element.tag] = virtual_element


class Header():
  """A dicom dataset blueprint for a factory to produce a series of dicom
  datasets"""
  def __getitem__(self, key) -> Union[DataElement, CallElement]:
    return self._blueprint[Tag(key)]

  def __setitem__(self, key, value: Union[DataElement, CallElement]):
    if key != value.tag:
      raise ValueError("Tag mismatch")
    self.add_tag(value)

  def __contains__(self, key) -> bool:
    return key in self._blueprint

  def __init__(self,
               blueprint: List[Union[DataElement, CallElement]] = [],
               ) -> None:
    self._blueprint: Dict[BaseTag,Union[DataElement, CallElement]] = {}
    for tag in blueprint:
      self.add_tag(tag)

  def add_tag(self, tag: Union[DataElement, CallElement]) -> None:
    if isinstance(tag, DataElement) or isinstance(tag, CallElement):
      self._blueprint[tag.tag] = tag
    else:
      raise InvalidTagType("Attempting to add a non instantiable tag to a header")

  def __iter__(self):
    for element in self._blueprint.values():
      yield element


class DicomFactory(ABC):
  """A DicomFactory is a class, that produces various collections of datasets
  """

  def __init__(self,
               header_blueprint: Optional[HeaderBlueprint] = None,
               filling_strategy: Optional[FillingStrategy] = FillingStrategy.DISCARD) -> None:
    self.header_blueprint: Optional[HeaderBlueprint] = header_blueprint
    self.filling_strategy: Optional[FillingStrategy] = filling_strategy
    self.series_description: str = "Unnamed Pipeline post processing "

  def make_header(self,
                  dataset: Dataset,
                  elements: Optional[HeaderBlueprint]= None,
                  filling_strategy: Optional[FillingStrategy] = None
    ) -> Header:
    """This function produces a header dataset based on an input dataset.

    Note that it's callers responsibility to ensure, that the produced header
    is uniform for all input datasets

    Args:
        dataset (Dataset): _description_

    Returns:
        Dataset: The produced dataset

    Raises:
        IncorrectlyConfigured: If it's impossible to produce a DataElement from a tag
    """
    if elements is None:
      elements = self.header_blueprint
    if elements is None:
      raise IncorrectlyConfigured("A header needs some tags")

    if filling_strategy is None:
      filling_strategy = self.filling_strategy
    if filling_strategy is None:
      raise IncorrectlyConfigured("A strategy is need for unmarked tags")

    header = Header()
    if filling_strategy == FillingStrategy.DISCARD:
      for virtual_element in elements:
        de = virtual_element.corporealialize(self, dataset)
        if de is not None:
          header.add_tag(de)
    elif filling_strategy == FillingStrategy.COPY:
      for data_element in dataset:
        if data_element.tag in elements:
          virtual_element = elements[data_element.tag]
          corporeal_tag = virtual_element.corporealialize(self, dataset)
          if corporeal_tag is not None:
            header.add_tag(corporeal_tag)
        else:
          header.add_tag(data_element)
    return header

  @abstractmethod
  def make_series(self,
                  header : Header,
                  image : Any
    ) -> List[Dataset]:
    raise NotImplementedError #pragma: no cover


class NoFactory(DicomFactory):
  def make_series(self, header: Header, image: Any) -> List[Dataset]:
    raise NotImplementedError

###### Header function ######
def _get_today() -> date:
  return date.today()

def _get_time() -> time:
  return datetime.now().time()

def _get_random_number() -> int:
  return randint(1, 2147483646)
###### Header Tag Lists ######

patient_header_tags = HeaderBlueprint([
  CopyElement(0x00100010), # PatientName
  CopyElement(0x00100020), # PatientID
  CopyElement(0x00100030), # PatientsBirthDate
  CopyElement(0x00100040), # PatientSex
])

general_study_header_tags = HeaderBlueprint([
  CopyElement(0x00080020), # StudyDate
  CopyElement(0x00080030), # StudyTime
  CopyElement(0x00080050), # AccessionNumber
  CopyElement(0x00081030, Optional=True), # StudyDescription
  CopyElement(0x00200010, Optional=True), # StudyID
  CopyElement(0x0020000D), # StudyInstanceUID
])

patient_study_header_tags = HeaderBlueprint([
  CopyElement(0x00101010, Optional=True), # PatientAge
  CopyElement(0x00101020, Optional=True), # PatientSize
  CopyElement(0x00101030, Optional=True), # PatientWeight
])


general_series_study_header = HeaderBlueprint([
  CopyElement(0x00080060), # Modality
  SeriesElement(0x00080021, 'DA', _get_today), # SeriesDate
  SeriesElement(0x00080031, 'TM', _get_time), # SeriesTime
  SeriesElement(0x0020000E, 'UI', gen_uid),   # SeriesInstanceUID
  AttrElement(0x0008103E, 'LO', 'series_description'),
  SeriesElement(0x00200011, 'IS', _get_random_number) # SeriesNumber
])
