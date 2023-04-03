""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard Library


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import Enum
from inspect import getfullargspec
from pprint import pformat
from random import randint
from typing import Any, Callable, Dict, List, Iterator,  Optional, Tuple, Union, Iterable

# Third Party Library
import numpy
from pydicom import DataElement, Dataset
from pydicom.tag import Tag, BaseTag

# Dicomnode Library
from dicomnode.lib.dicom import gen_uid
from dicomnode.lib.exceptions import InvalidTagType, HeaderConstructionFailure

class FillingStrategy(Enum):
  DISCARD = 0
  COPY = 1

class VirtualElement(ABC):
  """Represents an element in a blueprint.
  Each virtual element can produce a pydicom.DataElement
  """
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

  @property
  def VR(self) -> str:
    return self.__VR

  @VR.setter
  def VR(self, val: str) -> None:
    self.__VR = val

  def __init__(self, tag: Union[BaseTag, str, int, Tuple[int,int]], VR: str) -> None:
    self.tag = tag
    self.VR = VR

  def get_pivot(self, datasets: Iterable[Dataset]) -> Optional[Dataset]:
    dataset = None
    for _dataset in datasets:
      dataset = _dataset
      break
    return dataset

  @abstractmethod
  def corporealialize(self, factory: 'DicomFactory', dataset: Iterable[Dataset]) -> Optional[Union[DataElement, 'InstanceVirtualElement']]:
    raise NotImplemented # pragma: no cover

# Static Virtual Elements
class AttrElement(VirtualElement):
  """Reads an attribute from the factory and creates a data element
  from it"""
  def __init__(self, tag: Union[BaseTag, str, int, Tuple[int,int]], VR: str, attribute: str) -> None:
    super().__init__(tag, VR)
    self.attribute = attribute

  def corporealialize(self, factory: 'DicomFactory', _: Iterable[Dataset]) -> DataElement:
    value = getattr(factory, self.attribute)
    return DataElement(self.tag, self.VR, value)

class CopyElement(VirtualElement):
  """Virtual Data Element, indicating that the value will be copied from an
  original dataset, Throws an error is element is missing"""

  def __init__(self, tag: Union[BaseTag, str, int, Tuple[int,int]], Optional: bool = False) -> None:
    self.tag: BaseTag = Tag(tag)
    self.Optional = Optional

  def corporealialize(self, _factory: 'DicomFactory', datasets: Iterable[Dataset]) -> Optional[DataElement]:
    dataset = self.get_pivot(datasets)
    if dataset is None:
      error_message = f"Cannot copy nothing at tag:{self.tag}"
      raise ValueError(error_message)

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

  def corporealialize(self, _factory: 'DicomFactory', _dataset: Iterable[Dataset]) -> None:
    return None


class StaticElement(VirtualElement):
  def __init__(self, tag, VR, value) -> None:
    self.tag = tag
    self.VR = VR
    self.value = value

  def corporealialize(self, _factory: 'DicomFactory', _datasets: Iterable[Dataset]) -> DataElement:
    return DataElement(self.tag, self.VR, self.value)

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

  def corporealialize(self, _: 'DicomFactory', datasets: Iterable[Dataset]) -> DataElement:
    dataset = self.get_pivot(datasets)
    if self.require_dataset:
      # The '# type: ignores' here is because the type checker assumes func
      # is an instance based function, when it is dynamic assigned function
      # and therefore behaves like a static function
      value = self.func(dataset) # type: ignore
    else:
      value = self.func() # type: ignore
    return DataElement(self.tag, self.VR, value)

# End of static Virtual Elements
# InstanceBasedVirtual Elements

@dataclass
class InstanceEnvironment:
  instance_number: int
  header_dataset: Optional[Dataset] = None
  image: Optional[Any] = None # Unmodified image
  factory: Optional['DicomFactory'] = None
  intercept: Optional[float] = None
  slope: Optional[float] = None
  scaled_image: Optional[Any] = None
  total_images: Optional[int] = None

class InstanceVirtualElement(VirtualElement):
  """Represents a virtual element, that is unique per image slice"""

  def produce(self, instance_environment: InstanceEnvironment) -> DataElement:
    raise NotImplemented


class FunctionalElement(InstanceVirtualElement):
  """Abstract tag. This class represents a tag, that will be
  instantiated from with an image slice.
  """
  def __init__(self, tag: Union[BaseTag, str, int, Tuple[int,int]], VR: str, func: Callable[[InstanceEnvironment],Any]) -> None:
    super().__init__(tag, VR)
    self.func = func
    self.header_dataset: Optional[Dataset] = None

  def corporealialize(self, _factory: 'DicomFactory', dataset: Dataset) -> 'FunctionalElement':
    return self

  def produce(self, caller_args: InstanceEnvironment) -> Optional[DataElement]:
    value = self.func(caller_args)
    if value is None:
      return None
    return DataElement(self.tag, self.VR, value)


class InstanceCopyElement(InstanceVirtualElement):
  """This tag should be used to copy values from a series
  """
  def corporealialize(self, factory: 'DicomFactory', datasets: Iterable[Dataset]) -> 'InstanceCopyElement':
    for dataset in datasets:
      self.__instances[dataset.InstanceNumber] = dataset[self.tag].value
    return self

  def __init__(self, tag: Union[BaseTag, str, int, Tuple[int, int]], VR: str) -> None:
    super().__init__(tag, VR)
    self.__instances = {}

  def produce(self, instance_environment: InstanceEnvironment) -> DataElement:
    return DataElement(self.tag, self.VR, self.__instances[instance_environment.instance_number])


class Blueprint():
  """Blueprint for a dicom series. A blueprint contains no information on a specific series.
  """
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

  def __init__(self, virtual_elements: Union[List[VirtualElement],'Blueprint'] = []) -> None:
    # Init
    self._dict: Dict[int, VirtualElement] = {}

    # Fill from init
    for ve in virtual_elements:
      self.add_virtual_element(ve)

  def __len__(self) -> int:
    return len(self._dict)

  def __contains__(self, tag: int) -> bool:
    return tag in self._dict

  def __add__(self, blueprint: 'Blueprint') -> 'Blueprint':
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

  def get_required_tags(self) -> List[int]:
    return_list = []
    for virtual_element in self._dict.values():
      if isinstance(virtual_element, CopyElement):
        if virtual_element.Optional == False:
          return_list.append(int(virtual_element.tag))
      if isinstance(virtual_element, InstanceCopyElement):
        return_list.append(int(virtual_element.tag))
    return_list.sort()
    return return_list


class SeriesHeader():
  """Instantiated blueprint for a specific dicom series

  Can be used with raw image data to produce a dicom series
  """
  def __getitem__(self, key) -> Union[DataElement, InstanceVirtualElement]:
    return self._blueprint[Tag(key)]

  def __setitem__(self, key, value: Union[DataElement, InstanceVirtualElement]):
    if key != value.tag:
      raise ValueError("Tag mismatch")
    self.add_tag(value)

  def __contains__(self, key) -> bool:
    return key in self._blueprint

  def __init__(self,
               blueprint: List[Union[DataElement, InstanceVirtualElement]] = [],
               ) -> None:
    self._blueprint: Dict[BaseTag,Union[DataElement, InstanceVirtualElement]] = {}
    for tag in blueprint:
      self.add_tag(tag)

  def add_tag(self, tag: Union[DataElement, InstanceVirtualElement]) -> None:
    if isinstance(tag, DataElement) or isinstance(tag, InstanceVirtualElement):
      self._blueprint[tag.tag] = tag
    else:
      raise InvalidTagType("Attempting to add a non instantiable tag to a header")

  def __iter__(self):
    for element in self._blueprint.values():
      yield element

  def __str__(self) -> str:
    message = f"SeriesHeader with {len(self._blueprint)} tags:\n"
    for tag in self:
      message += f"    Tag: {tag.tag} - {tag.__class__.__name__}\n"

    return message

class DicomFactory(ABC):
  """A DicomFactory produces Series of Dicom Datasets and everything needed to produce them.

  This is a base class, as factories are specialized per image input type
  """

  def __init__(self) -> None:

    self.series_description: str = "Unnamed Pipeline post processing "

  def make_series_header(self,
                  pivot_list: List[Dataset],
                  blueprint: Blueprint,
                  filling_strategy: FillingStrategy = FillingStrategy.DISCARD
    ) -> SeriesHeader:
    """This function produces a header dataset based on an input datasets.

    Note: There's a tutorial for creating Headers at:
    https://github.com/Rigshospitalet-KFNM/DicomNode/tutorials/MakingHeaders.md

    Args:
        pivot (Dataset): The dataset which the header will be produced from

    Returns:
        SeriesHeader: This object is a "header" for the series
    """
    failed_tags = []
    header = SeriesHeader()
    if len(pivot_list) == 0:
      raise ValueError("Cannot create header without a pivot dataset")
    pivot = pivot_list[0]

    if filling_strategy == FillingStrategy.COPY:
      for data_element in pivot:
        if data_element.tag in blueprint:
          pass # Will be added in
        else:
          header.add_tag(data_element)
    for virtual_element in blueprint:
      try:
        de = virtual_element.corporealialize(self, pivot_list)
        if de is not None:
          header.add_tag(de)
      except KeyError:
        failed_tags.append(virtual_element.tag)
    if len(failed_tags) != 0:
      error_message = f"Pivot is missing: {pformat(failed_tags)}"
      raise HeaderConstructionFailure(error_message)

    return header

  @abstractmethod
  def build_from_header(self,
                  header : SeriesHeader,
                  image : Any
    ) -> List[Dataset]:
    raise NotImplementedError #pragma: no cover

  def build(self, pivot: Dataset, blueprint: Blueprint, filling_strategy: FillingStrategy = FillingStrategy.DISCARD) -> Dataset:
    """Builds a singular dataset from blueprint and pivot dataset

    Intended to be used to construct message datasets

    Args:
        pivot (Dataset): Dataset the blueprint will use to extract data from
        blueprint (Blueprint): Determines what data will be in the newly construct dataset
        filling_strategy (FillingStrategy, optional): strategy to handle tags in the dataset,
        but in the blueprint. Defaults to FillingStrategy.DISCARD,
        and for most dataset this is the sensible option

    Returns:
        Dataset: The constructed dataset
    """
    failed_tags = []
    dataset = Dataset()
    if filling_strategy == FillingStrategy.COPY:
      for data_element in pivot:
        if data_element.tag in blueprint:
          pass
        else:
          dataset.add(data_element)
    for virtual_element in blueprint:
      de = virtual_element.corporealialize(self, [pivot])
      if isinstance(de, InstanceVirtualElement):
        args = InstanceEnvironment(1)
        de = de.produce(args)
      if de is not None:
        dataset.add(de)
    return dataset

###### Header function ######

def _add_InstanceNumber(caller_args: InstanceEnvironment):
  # iterator is Zero indexed while, instance number is 1 indexed
  # This function assumes that the factory is aware of this
  return caller_args.instance_number

def _add_SOPInstanceUID(_):
  return gen_uid()

def _get_today(_) -> date:
  return date.today()

def _get_time(_) -> time:
  return datetime.now().time()

def _get_random_number(_) -> int:
  return randint(1, 2147483646)

###### Header Tag Lists ######

patient_blueprint = Blueprint([
  CopyElement(0x00100010), # PatientName
  CopyElement(0x00100020), # PatientID
  CopyElement(0x00100021, Optional=True), # Issuer of Patient ID
  CopyElement(0x00100030, Optional=True), # PatientsBirthDate
  CopyElement(0x00100040), # PatientSex
])

frame_of_reference_blueprint = Blueprint([
  CopyElement(0x00200052),
  CopyElement(0x00201040)
])

general_study_blueprint = Blueprint([
  CopyElement(0x00080020), # StudyDate
  CopyElement(0x00080030), # StudyTime
  CopyElement(0x00080050), # AccessionNumber
  CopyElement(0x00081030, Optional=True), # StudyDescription
  CopyElement(0x00200010, Optional=True), # StudyID
  CopyElement(0x0020000D), # StudyInstanceUID
])

# You might argue that you should overwrite, since this is a synthetic image
general_equipment_blueprint = Blueprint([
  CopyElement(0x00080070, Optional=True), # Manufacturer
  CopyElement(0x00080080, Optional=True), # Institution Name
  CopyElement(0x00080081, Optional=True), # Institution Address
  CopyElement(0x00081040, Optional=True), # Institution Department Name
  CopyElement(0x00081090, Optional=True), # Manufacturer's Model Name
])

general_image_blueprint = Blueprint([
  StaticElement(0x00080008, 'CS', ['DERIVED', 'PRIMARY']), # Image Type # write a test for this
  FunctionalElement(0x00200013, 'IS', _add_InstanceNumber), # InstanceNumber

])

# One might argue the optionality of these tags
image_plane_blueprint = Blueprint([
  CopyElement(0x00180050),               # Slice thickness
  InstanceCopyElement(0x00200032, 'DS'), # Image position
  CopyElement(0x00200037),               # Image Orientation
  #InstanceCopyElement(0x00201041, 'DS'), # Slice Location
  CopyElement(0x00280030),               # Pixel Spacing
])

ct_image_blueprint = Blueprint([
  CopyElement(0x00080008, Optional=True), # Image Type
  CopyElement(0x00180022, Optional=True), # Scan Options
  CopyElement(0x00180060, Optional=True), # KVP
  CopyElement(0x00180090, Optional=True), # Data Collection Diameter
  CopyElement(0x00181100, Optional=True), # Reconstruction Diameter
  CopyElement(0x00181110, Optional=True), # Distance Source to Detector
  CopyElement(0x00181111, Optional=True), # Distance Source to Patient
  CopyElement(0x00181120, Optional=True), # Gantry / Detector Tilt
  CopyElement(0x00181130, Optional=True), # Table Height
  CopyElement(0x00181140, Optional=True), # Rotation Direction
  CopyElement(0x00181150, Optional=True), # Exposure Time
  CopyElement(0x00181151, Optional=True), # X-Ray Tube Current
  CopyElement(0x00181152, Optional=True), # Exposure
  CopyElement(0x00181153, Optional=True), # Exposure in µAs
  CopyElement(0x00181160, Optional=True), # Filter Type
  CopyElement(0x00181170, Optional=True), # Generator Power
  CopyElement(0x00181190, Optional=True), # Focal Spots
  CopyElement(0x00181210, Optional=True), # Convolution Kernel
  CopyElement(0x00189305, Optional=True), # Revolution Time
])


patient_study_blueprint = Blueprint([
  CopyElement(0x00101010, Optional=True), # PatientAge
  CopyElement(0x00101020, Optional=True), # PatientSize
  CopyElement(0x00101022, Optional=True), # PatientBodyMassIndex
  CopyElement(0x00101030, Optional=True), # PatientWeight
  CopyElement(0x001021A0, Optional=True), # SmokingStatus
  CopyElement(0x001021C0, Optional=True), # PregnancyStatus
])


general_series_blueprint = Blueprint([
  CopyElement(0x00080060), # Modality
  SeriesElement(0x00080021, 'DA', _get_today),         # SeriesDate
  SeriesElement(0x00080031, 'TM', _get_time),          # SeriesTime
  SeriesElement(0x0020000E, 'UI', gen_uid),            # SeriesInstanceUID
  AttrElement(0x0008103E, 'LO', 'series_description'), # SeriesDescription
  SeriesElement(0x00200011, 'IS', _get_random_number), # SeriesNumber
  CopyElement(0x00081070, Optional=True),              # Operators' Name
  CopyElement(0x00185100),                             # PatientPosition
])

SOP_common_blueprint: Blueprint = Blueprint([
  CopyElement(0x00080016),                                  # SOPClassUID, you might need to change this
  FunctionalElement(0x00080018, 'UI', _add_SOPInstanceUID), # SOPInstanceUID
  FunctionalElement(0x00200013, 'IS', _add_InstanceNumber)  # InstanceNumber
])
