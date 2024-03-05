""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard Library
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import Enum
from inspect import getfullargspec
from pathlib import Path
from pprint import pformat
from random import randint
from typing import Any, Callable, Dict, Generic, List, Iterator, Iterable,  Optional, Tuple, TypeVar, Union

# Third Party Library
from pydicom import DataElement, Dataset, Sequence
from pydicom.uid import EncapsulatedPDFStorage
from pydicom.tag import Tag, BaseTag
from sortedcontainers import SortedDict

# Dicomnode Library
from dicomnode.lib.exceptions import IncorrectlyConfigured
from dicomnode.lib.logging import get_logger
from dicomnode.dicom import gen_uid
from dicomnode.lib.exceptions import InvalidTagType, HeaderConstructionFailure

logger = get_logger()

T = TypeVar('T')

class FillingStrategy(Enum):
  DISCARD = 0
  COPY = 1

PRIVATIZATION_VERSION = 1

def get_pivot(datasets: Union[Dataset,Iterable[Dataset]]) -> Optional[Dataset]:
  if isinstance(datasets, Dataset):
    return datasets
  dataset = None
  for _dataset in datasets:
    dataset = _dataset
    break
  return dataset

class Reserved_Tags(Enum):
  PRIVATE_TAG_NAMES = 0xFE
  PRIVATE_TAG_VRS = 0xFF

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

  @property
  def is_private(self) -> bool:
    return bool((self.tag >> 16) % 2)

  @VR.setter
  def VR(self, val: str) -> None:
    self.__VR = val

  def __init__(self,
               tag: Union[BaseTag, str, int, Tuple[int,int]],
               VR: str,
               name: Optional[str]=None) -> None:
    self.tag = tag
    self.VR = VR
    if name is None:
      self.name = DataElement(tag, VR, None).name
    elif 64 < len(name):
      logger.warning(f"Virtual Element name is being truncated from {name} to {name[64:]}")
    else:
      self.name = name[64:]

  @abstractmethod
  def corporealialize(self,
                      factory: 'DicomFactory',
                      parent_datasets: Iterable[Dataset]
                      ) -> Optional[Union[DataElement,
                                          'InstanceVirtualElement']]:
    """Extracts data from the parent datasets and either produces a static
    element or a InstancedVirtualElement, in the case of that the produced tag
    should vary image instance to image instance.

    Args:
      factory (DicomFactory): Factory that's producing the series header
      parent_datasets (Iterable[Dataset]): Parent datasets to be extracted

    """
    raise NotImplemented # pragma: no cover



class CopyElement(VirtualElement):
  """Virtual Data Element, indicating that the value will be copied from an
  original dataset, Throws an error is element is missing"""

  def __init__(self, tag: Union[BaseTag, str, int, Tuple[int,int]], Optional: bool = False, name: Optional[str] = None) -> None:
    super().__init__(tag, '', name)
    self.Optional = Optional

  def corporealialize(self, _factory: 'DicomFactory', datasets: Iterable[Dataset]) -> Optional[DataElement]:
    dataset = get_pivot(datasets)
    if dataset is None:
      error_message = f"Cannot copy nothing at tag:{self.tag}"
      raise ValueError(error_message)

    if self.tag in dataset:
      return dataset[self.tag]
    else:
      if self.Optional:
        return None
      else:
        raise KeyError(f"{self.tag} - {self.name}: not found in Header Parent Dataset")


class DiscardElement(VirtualElement):
  def __init__(self, tag) -> None:
    self.tag = tag

  def corporealialize(self, _factory: 'DicomFactory', _dataset: Iterable[Dataset]) -> None:
    return None


class StaticElement(VirtualElement, Generic[T]):
  def __init__(self,
               tag: Union[str, BaseTag, int, Tuple[int, int]],
               VR: str, value: T,
               name:Optional[str]=None) -> None:
    super().__init__(tag, VR, name=name)
    self.value: T = value

  def corporealialize(self, _factory: 'DicomFactory', _datasets: Iterable[Dataset]) -> DataElement:
    return DataElement(self.tag, self.VR, self.value)

  def __str__(self):
    return f"<StaticElement: {self.tag} {self.VR} {self.value}>"

class SeriesElement(VirtualElement):
  """This virtual element is instantiated when the header is created
  It may depend on input dataset"""
  def __init__(self,
               tag: Union[BaseTag, str, int, Tuple[int,int]],
               VR: str,
               func: Union[Callable[[Dataset], Any], Callable[[],Any]],
               name=None) -> None:
    super().__init__(tag, VR, name=name)
    self.func:  Union[Callable[[Dataset], Any], Callable[[],Any]] = func
    self.require_dataset: bool = len(getfullargspec(func).args) == 1

  def corporealialize(self, _: 'DicomFactory', datasets: Iterable[Dataset]) -> DataElement:
    dataset = get_pivot(datasets)
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

@dataclass(slots=True)
class InstanceEnvironment:
  instance_number: int
  kwargs : Dict[Any, Any] = field(default_factory=dict)
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
    raise NotImplemented # pragma: no cover

class SequenceElement(InstanceVirtualElement):
  def __init__(self,
               tag: Union[BaseTag,str, int, Tuple[int, int]],
               sequence_blueprints: Iterable['Blueprint'],
               name: Optional[str] = None) -> None:
    super().__init__(tag, 'SQ', name)

    self._blueprints = sequence_blueprints
    self._partial_initialized_sequences: Optional[List[List[
      Union['InstanceVirtualElement', DataElement]]]] = None

  def corporealialize(self,
                      factory: 'DicomFactory',
                      parent_datasets: Iterable[Dataset]) -> 'SequenceElement':
    self._partial_initialized_sequences = []
    for blueprint in self._blueprints:
      partial_initialized_sequence = []
      for virtual_element in blueprint:
        corporealialize_value = virtual_element.corporealialize(factory, parent_datasets)
        if corporealialize_value is not None:
          partial_initialized_sequence.append(corporealialize_value)
      self._partial_initialized_sequences.append(partial_initialized_sequence)

    return self

  def produce(self, instance_environment: InstanceEnvironment) -> DataElement:
    if self._partial_initialized_sequences is None:
      logger.error("You are attempting to produce from an uninitialized sequence element")
      raise Exception

    sequence_datasets = []
    for partial_initialized_sequence in self._partial_initialized_sequences:
      sequence_dataset = Dataset()
      for partial_initialized_data_element in partial_initialized_sequence:
        if isinstance(partial_initialized_data_element, DataElement):
          sequence_dataset.add(partial_initialized_data_element)
        else:
          sequence_dataset.add(partial_initialized_data_element.produce(instance_environment))
      sequence_datasets.append(sequence_dataset)

    return DataElement(self.tag, 'SQ', Sequence(sequence_datasets))


class FunctionalElement(InstanceVirtualElement):
  """Abstract tag. This class represents a tag, that will be
  instantiated from with an image slice.
  """
  def __init__(self,
               tag: Union[BaseTag, str, int, Tuple[int,int]],
               VR: str,
               func: Callable[[InstanceEnvironment],Any],
               name: Optional[str]= None) -> None:
    super().__init__(tag, VR, name=name)
    self.func = func
    self.header_dataset: Optional[Dataset] = None

  def corporealialize(self, _factory: 'DicomFactory', dataset: Dataset) -> 'FunctionalElement':
    return self

  def produce(self, caller_args: InstanceEnvironment) -> Optional[DataElement]:
    return DataElement(self.tag, self.VR, self.func(caller_args))


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
  def __getitem__(self, tag: int) -> VirtualElement:
    return self._dict[tag]

  def __setitem__(self, tag: int, virtual_element: VirtualElement) -> None:
    if not isinstance(tag, BaseTag):
      tag = Tag(tag)
    if not isinstance(virtual_element, VirtualElement):
      raise TypeError("HeaderBlueprint only accepts VirtualElements as element")
    if tag != virtual_element.tag:
      raise ValueError("Miss match between tag and VirtualElement's tag")
    self.add_virtual_element(virtual_element)

  def __delitem__(self, tag):
    del self._dict[tag]

  def __init__(self, virtual_elements: Union[List[VirtualElement],'Blueprint'] = []) -> None:
    # Init
    self._dict: SortedDict = SortedDict()

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
      # https://github.com/grantjenks/python-sortedcontainers/pull/107
      # # YAY open source
      # If wanna fix this
      # https://github.com/althonos/python-sortedcontainers
      yield ve # type: ignore

  def add_virtual_element(self, virtual_element: VirtualElement):
    tag = virtual_element.tag
    if virtual_element.is_private:
      if not self.__validatePrivateTags(virtual_element):
        # Logging was done by validate tags
        raise IncorrectlyConfigured
      group_id = (tag & 0xFFFF0000) + ((tag >> 8) & 0xFF)
      tag_group = tag & 0xFFFFFF00
      index = len(list(self._dict.irange(tag_group, tag))) - 1
      tag_name = tag_group + Reserved_Tags.PRIVATE_TAG_NAMES.value
      tag_VR = tag_group + Reserved_Tags.PRIVATE_TAG_VRS.value

      # Check if the Group is allocated

      if not group_id in self:
        self._dict[group_id] = StaticElement[str](group_id, "LO", f"Dicomnode - Private tags version: {PRIVATIZATION_VERSION}")
        self._dict[tag_name] = StaticElement[List[str]](tag_name, "LO", [])
        self._dict[tag_VR] = StaticElement[List[str]](tag_VR, "LO", [])

      if tag in self:
        for reserved_tag_header in Reserved_Tags:
          static_element: StaticElement[List[str]] = self[tag_group + reserved_tag_header.value] # type: ignore
          del static_element.value[index]

      VE_names: StaticElement[List[str]] = self._dict[tag_name]
      VE_names.value.insert(index, virtual_element.name)
      VE_VR: StaticElement[List[str]] = self._dict[tag_VR]
      VE_VR.value.insert(index, virtual_element.VR)

      # End private tag handling
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

  def __validatePrivateTags(self, virtual_element: VirtualElement) -> bool:
    ALLOCATE_TAG_FILTER = 0xFF00
    tags_is_an_allocator = not bool(virtual_element.tag & ALLOCATE_TAG_FILTER)

    if tags_is_an_allocator:
      logger.error("Dicom node will automatically allocate private tag ranges")
      return False

    subTag = virtual_element.tag & 0xFF
    for reserved_tag in Reserved_Tags:
      if subTag == reserved_tag.value:
        logger.error("You are trying to add a private tag, that have been reserved by Dicomnode")
        return False

    return True

###### Header function ######

def _add_InstanceNumber(caller_args: InstanceEnvironment):
  # iterator is Zero indexed while, instance number is 1 indexed
  # This function assumes that the factory is aware of this
  return caller_args.instance_number

def _add_UID_tag(_: InstanceEnvironment):
  return gen_uid()

def _get_today(_: InstanceEnvironment) -> date:
  return date.today()

def _get_time(_: InstanceEnvironment) -> time:
  return datetime.now().time()

def _get_now_datetime(_: InstanceEnvironment) -> datetime:
  return datetime.now()

def _get_random_number(_:InstanceEnvironment) -> int:
  return randint(1, 2147483646)


default_report_blueprint = Blueprint([
  CopyElement(0x00080020, Optional=True), # Study Date
  FunctionalElement(0x00080023, 'DA', _get_today), #ContentDate
  FunctionalElement(0x0008002A, 'DT', _get_now_datetime), # AcquisitionDateTime
  FunctionalElement(0x00080033, 'TM', _get_time), #ContentTime
  CopyElement(0x00080030, Optional=True), # Study Time
  CopyElement(0x00080050), # AccessionNumber
  StaticElement(0x00080060, 'CS', 'DOC'), # Modality
  StaticElement(0x00080064, 'CS', 'WSD'), # ConversionType
  CopyElement(0x00081030), # StudyDescription
  CopyElement(0x00101010, Optional=True), #PatientAge
  CopyElement(0x00101020, Optional=True), #PatientSize
  CopyElement(0x00101030, Optional=True), #PatientWeight
  CopyElement(0x0020000D), # StudyInstanceUID
  CopyElement(0x00200010, Optional=True), # StudyID
  StaticElement(0x00200012, 'IS', 1), # InstanceNumber
  SeriesElement(0x0020000E, 'UI', _add_UID_tag), # SeriesInstanceUID
  StaticElement(0x00280301, 'CS', 'NO'),  # BurnedInAnnotation
])


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


  def get_default_blueprint(self) -> Blueprint:
    return Blueprint()

  def make_series_header(self,
                  pivot_list: List[Dataset],
                  blueprint: Blueprint,
                  filling_strategy: FillingStrategy = FillingStrategy.DISCARD,
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
                  image : Any,
                  kwargs : Dict[Any, Any] = {}
    ) -> List[Dataset]:
    raise NotImplementedError #pragma: no cover

  def build(self,
            pivot: Dataset,
            blueprint: Blueprint,
            filling_strategy: FillingStrategy = FillingStrategy.DISCARD,
            kwargs: Dict[Any, Any] = {}) -> Dataset:
    """Builds a singular dataset from blueprint and pivot dataset

    Intended to be used to singular such as messages or report datasets

    Args:
        pivot (Dataset): Dataset the blueprint will use to extract data from
        blueprint (Blueprint): Determines what data will be in the newly
                               constructed dataset
        filling_strategy (FillingStrategy, optional): Strategy to handle tags
                                                      that in the dataset, but
                                                      in not the blueprint.
                                                      Discards ignores the tag,
                                                      while copies the
                                                      un-annotated tag. Defaults
                                                      to FillingStrategy.DISCARD,
                                                      and for most dataset this
                                                      is the sensible option

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
        args = InstanceEnvironment(1, kwargs=kwargs)
        de = de.produce(args)
      if de is not None:
        dataset.add(de)
    return dataset

  def encode_pdf(self,
                 report: Union['Report', Path, str],
                 datasets: Iterable[Dataset],
                 report_blueprint = default_report_blueprint,
                 filling_strategy = FillingStrategy.DISCARD,
                 kwargs: Dict[Any, Any] = {}) -> Dataset:
    """Encodes a pdf file to a dicom file such that it can be transported with
    the Dicom protocol

    Args:
        report (Union[Report, Path, str]): The report or a path to the PDF to be
        encoded
        datasets (Iterable[Dataset]): The datasets that have been used to
        generate this report
        report_blueprint (Blueprint, optional): The dicomnode.Blueprint for
        the generated report. The Blueprint should not include the tags:
          EncapsulatedDocument, EncapsulatedDocumentLength, SourceInstanceSequence
        Defaults to default_report_blueprint.
        filling_strategy (dicomnode.FillingStrategy, optional): Filling
        strategy for the underlying build command. For more detailed docs look
        at the build method. Defaults to FillingStrategy.DISCARD.
        kwargs (Dict[Any, Any], optional): Extra kwargs for underlying build
        command. Defaults to {}.

    Returns:
        Dataset: An EncapsulatedPDFStorage encoded Dicom Dataset with the input
        report
    """
    from dicomnode.report import Report
    # Getting the PDF data
    if isinstance(report, Report):
      report.generate_pdf()
      report_file_path = Path(report.file_name + '.pdf')
    elif isinstance(report, str):
      if not report.endswith('.pdf'):
        logger.error("Passed a none pdf file to encoding")
      report_file_path = Path(report)
    else:
      report_file_path = report

    with open(report_file_path, 'rb') as fp:
      document_bytes = fp.read()

    pivot = get_pivot(datasets)
    report_dataset = self.build(pivot, report_blueprint, filling_strategy, kwargs)

    if 'EncapsulatedDocument' not in report_dataset:
      report_dataset.EncapsulatedDocument = document_bytes
      report_dataset.EncapsulatedDocumentLength = len(document_bytes)
    else:
      logger.warning("report dataset already have a encoded PDF document from build phase")

    if 'SOPClassUID' not in report_dataset:
      report_dataset.SOPClassUID = EncapsulatedPDFStorage
    if 'Modality' not in report_dataset:
      report_dataset.Modality = "DOC"
    if 'ConversionType' not in report_dataset:
      report_dataset.ConversionType = "WSD"
    if 'MIMETypeOfEncapsulatedDocument' not in report_dataset:
      report_dataset.MIMETypeOfEncapsulatedDocument = "application/pdf"
    if 'SourceInstanceSequence' not in report_dataset:
      sequence = []
      for reference_dataset in datasets:
        if 'SOPClassUID' in reference_dataset and 'SOPInstanceUID' in reference_dataset:
           sequence_dataset = Dataset()
           sequence_dataset.ReferencedSOPClassUID = reference_dataset.SOPClassUID
           sequence_dataset.ReferencedSOPInstanceUID = reference_dataset.SOPInstanceUID
           sequence.append(sequence_dataset)
      report_dataset.SourceInstanceSequence = Sequence(sequence)

    return report_dataset


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
  SeriesElement(0x0008103E, 'LO', lambda: "Dicomnode pipeline output"), # SeriesDescription
  SeriesElement(0x00200011, 'IS', _get_random_number), # SeriesNumber
  CopyElement(0x00081070, Optional=True),              # Operators' Name
  CopyElement(0x00185100),                             # PatientPosition
])

SOP_common_blueprint: Blueprint = Blueprint([
  CopyElement(0x00080016),                                  # SOPClassUID, you might need to change this
  FunctionalElement(0x00080018, 'UI', _add_UID_tag), # SOPInstanceUID
  FunctionalElement(0x00200013, 'IS', _add_InstanceNumber)  # InstanceNumber
])

