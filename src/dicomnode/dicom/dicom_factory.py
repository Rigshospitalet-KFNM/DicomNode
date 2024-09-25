""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard Library
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import Enum
from inspect import getfullargspec
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Iterator, Iterable,\
    Optional,Sequence as TypingSequence, Tuple, TypeVar, Union

# Third Party Library
from numpy import ndarray, zeros_like
from pydicom import DataElement, Dataset, Sequence
from pydicom.uid import EncapsulatedPDFStorage, SecondaryCaptureImageStorage
from pydicom.tag import Tag, BaseTag
from sortedcontainers import SortedDict

# Dicomnode Library
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING, SIGNED_ARRAY_ENCODING
from dicomnode.dicom import make_meta
from dicomnode.dicom.series import DicomSeries
from dicomnode.math.image import fit_image_into_unsigned_bit_range
from dicomnode.lib.exceptions import IncorrectlyConfigured, InvalidDataset, MissingOptionalDependency
from dicomnode.lib.logging import get_logger
from dicomnode.lib.exceptions import MissingPivotDataset

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
  def corporealialize(self, datasets: Iterable[Dataset]
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

  def corporealialize(self, datasets: Iterable[Dataset]) -> Optional[DataElement]:
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

  def corporealialize(self, _datasets: Iterable[Dataset]) -> None:
    return None


class StaticElement(VirtualElement, Generic[T]):
  def __init__(self,
               tag: Union[str, BaseTag, int, Tuple[int, int]],
               VR: str, value: T,
               name:Optional[str]=None) -> None:
    super().__init__(tag, VR, name=name)
    self.value: T = value

  def corporealialize(self, _datasets: Iterable[Dataset]) -> DataElement:
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

  def corporealialize(self, datasets: Iterable[Dataset]) -> DataElement:
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

@dataclass
class InstanceEnvironment:
  instance_number: int
  factory: 'DicomFactory'
  kwargs : Dict[Any, Any] = field(default_factory=dict)
  instance_dataset: Optional[Dataset] = None
  image: Optional[ndarray[Tuple[int,int,int], Any]] = None # Unmodified image

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
                      parent_datasets: Iterable[Dataset]) -> 'SequenceElement':
    self._partial_initialized_sequences = []
    for blueprint in self._blueprints:
      partial_initialized_sequence = []
      for virtual_element in blueprint:
        corporealialize_value = virtual_element.corporealialize(parent_datasets)
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

  def corporealialize(self, datasets: Iterable[Dataset]) -> 'FunctionalElement':
    return self

  def produce(self, caller_args: InstanceEnvironment) -> Optional[DataElement]:
    return DataElement(self.tag, self.VR, self.func(caller_args))


class InstanceCopyElement(InstanceVirtualElement):
  """This tag should be used to copy values from a series
  """
  def corporealialize(self, datasets: Iterable[Dataset]) -> 'InstanceCopyElement':
    return self

  def __init__(self,
               tag: Union[BaseTag, str, int, Tuple[int, int]],
               VR: str,
               name=None
              ) -> None:
    super().__init__(tag, VR, name=name)

  def produce(self, instance_environment: InstanceEnvironment) -> DataElement:
    if instance_environment.instance_dataset is None:
      raise InvalidDataset

    return instance_environment.instance_dataset.get(self.tag,
                                                     DataElement(self.tag, self.VR, None))

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

  def __init__(self, virtual_elements: Union[TypingSequence[VirtualElement],'Blueprint'] = []) -> None:
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

class DicomFactory():
  """A DicomFactory produces Series of Dicom Datasets and everything needed to produce them.

  This is a base class, as factories are specialized per image input type
  """

  class PixelRepresentation(Enum):
    UNSIGNED = 0
    TWOS_COMPLIMENT = 1

  class PhotometricInterpretation(Enum):
    MONOCHROME_WHITE = "MONOCHROME1"
    """MONOCHROME1, the minimum sample value is intended to be displayed as
    white
    """
    MONOCHROME_BLACK = "MONOCHROME2"
    """MONOCHROME1, the minimum sample value is intended to be displayed as
    black
    """

  def __init__(self) -> None:
    # Default properties
    self.default_bits_stored = 16
    self.default_bits_allocated = 16
    self.default_pixel_representation = DicomFactory.PixelRepresentation.UNSIGNED
    self.default_photometric_interpretation = DicomFactory.PhotometricInterpretation.MONOCHROME_BLACK

  def store_image_in_dataset(self, dataset: Dataset, image: ndarray[Tuple[int,int], Any]):
    if len(image.shape) != 2:
      raise ValueError("You can only store an 2D image using this function")

    if self.default_pixel_representation == self.PixelRepresentation.UNSIGNED:
      stored_image_type = UNSIGNED_ARRAY_ENCODING.get(self.default_bits_allocated, None)
    elif self.default_pixel_representation == self.PixelRepresentation.TWOS_COMPLIMENT:
      raise NotImplemented("Signed encoding is not supported yet")
      #stored_image_type = SIGNED_ARRAY_ENCODING.get(self.default_bits_allocated, None)
    if stored_image_type is None:
      raise IncorrectlyConfigured("default bits allocated must be 8,16,32,64")

    if stored_image_type != image.dtype:
      encoded_image, slope, intercept = fit_image_into_unsigned_bit_range(image,
                                                                          self.default_bits_stored,
                                                                          self.default_bits_allocated)
      store_rescale = True
    else:
      encoded_image, slope, intercept = (image, 1, 0)
      store_rescale = False

    if 0x00080016 not in dataset:
      dataset.SOPClassUID = SecondaryCaptureImageStorage # Sets this temporary
    make_meta(dataset)

    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = self.default_photometric_interpretation.value
    dataset.Columns = encoded_image.shape[1]
    dataset.Rows = encoded_image.shape[0]
    dataset.BitsAllocated = self.default_bits_allocated
    dataset.BitsStored = self.default_bits_stored
    dataset.HighBit = self.default_bits_stored - 1
    dataset.SmallestImagePixelValue = int(encoded_image.min())
    dataset.LargestImagePixelValue = int(encoded_image.max())
    dataset.PixelRepresentation = self.default_pixel_representation.value

    if store_rescale:
      dataset.RescaleSlope = slope
      dataset.RescaleIntercept = intercept

    dataset.PixelData = encoded_image.tobytes()


  def build_series(self,
                   image: ndarray[Tuple[int,int,int],Any],
                   blueprint: Blueprint,
                   parent_series: Union[DicomSeries, List[Dataset]],
                   filling_strategy: FillingStrategy = FillingStrategy.DISCARD,
                   kwargs: Dict[Any, Any] = {}
                   ) -> DicomSeries:
    """Builds a dicom series from a series, from an image, blueprint, and series

    The following tags are always added:
      (0028,0002) - Samples per Pixel
      (0028,0004) - Photometric Interpretation
      (0028,0010) - Rows
      (0028,0011) - Columns
      (0028,0100) - Bit Allocated
      (0028,0101) - Bit Stored
      (0028,0102) - High bit
      (0028,0103) - Pixel Representation
      (7FE0,0010) - Pixel Data

    Args:
        image (ndarray): _description_
        blueprint (Blueprint): _description_
        parent_series (Series): _description_
        filling_strategy (FillingStrategy, optional): _description_. Defaults to FillingStrategy.DISCARD.
        kwargs (Dict[Any, Any], optional): _description_. Defaults to {}.

    Returns:
        Series: _description_
    """

    # Dataset creation have been difficult to design
    #
    # Originally this was solved in a two step process, where the static data
    # were produced first then the series was build secondly
    #
    # However this caused some really big problems with regards producing
    # multiple series
    #
    # The ways that it's solved now is that most things are wrapped in a series
    #
    if not len(image.shape) == 3:
      raise ValueError("Image must be a 3 dimensional image")

    if isinstance(parent_series, List):
      parent_datasets = parent_series
      can_copy_instances = len(parent_datasets) == image.shape[2]
    else:
      parent_datasets = parent_series.datasets
      can_copy_instances = parent_series.can_copy_into_image(image)


    datasets = []

    for i, image_slice in enumerate(image):
      dataset = Dataset()
      self.store_image_in_dataset(dataset, image_slice)
      dataset.InstanceNumber = i + 1

      datasets.append(dataset)

    series = DicomSeries(datasets)

    for virtual_tag in blueprint:
      result = virtual_tag.corporealialize(parent_datasets)
      if isinstance(result, DataElement):
        series.set_shared_tag(virtual_tag.tag, result)
      elif isinstance(result, InstanceVirtualElement):
        if can_copy_instances:
          envs = [
            InstanceEnvironment(
              instance_number=i,
              factory=self,
              kwargs=kwargs,
              instance_dataset=dataset,
              image=image,
            ) for (i, dataset) in enumerate(parent_datasets)
          ]
        else:
          envs = [
            InstanceEnvironment(
              instance_number=i,
              factory=self,
              kwargs=kwargs,
              image=image,
            ) for i in range(image.shape[0])
          ]

        data_elements = [
          result.produce(env) for env in envs
        ]

        series.set_individual_tag(virtual_tag.tag,
                                  data_elements)

    for dataset in series.datasets:
      make_meta(dataset)

    return series

  def build_instance(self,
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

    dataset = Dataset()
    if filling_strategy == FillingStrategy.COPY:
      for data_element in pivot:
        if data_element.tag in blueprint:
          pass
        else:
          dataset.add(data_element)
    for virtual_element in blueprint:
      de = virtual_element.corporealialize([pivot])
      if isinstance(de, InstanceVirtualElement):
        args = InstanceEnvironment(1, self, kwargs=kwargs)
        de = de.produce(args)
      if de is not None:
        dataset.add(de)
    if 0x0008_0016 not in dataset:
      raise ValueError("You are attempting to create a Dataset with out an SOPClassUID")
    make_meta(dataset)
    return dataset

  def encode_pdf(self,
                 report: Union['Report', Path, str], # type:ignore
                 datasets: Iterable[Dataset],
                 report_blueprint: Blueprint,
                 filling_strategy=FillingStrategy.DISCARD,
                 as_secondary_image_capture=False,
                 kwargs: Dict[Any, Any] = {}) -> List[Dataset]:
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

    Raises:
        MissingPivotDataset: If unable to get a dataset from datasets argument
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

    report_class_UID = None

    if 0x0008_0016 in report_blueprint:
      class_UID_element = report_blueprint[0x0008_0016]
      if isinstance(class_UID_element, StaticElement):
        report_class_UID = class_UID_element.value

    if report_class_UID is None:
      raise ValueError("Unable to determine class UID of dicom series report")

    handler_function = self.REPORT_HANDLER_FUNCTIONS.get(report_class_UID, None)

    if handler_function is not None:
      return handler_function(datasets,
                              report_blueprint,
                              filling_strategy,
                              document_bytes,
                              kwargs)
    raise ValueError(f"No handler function for {report_class_UID}")

  def _build_pdf_in_encapsulated_document(self,
                                          datasets,
                                          report_blueprint,
                                          filling_strategy,
                                          document_bytes,
                                          kwargs) -> List[Dataset]:
    pivot = get_pivot(datasets)
    if pivot is None:
      raise MissingPivotDataset
    report_dataset = self.build_instance(pivot, report_blueprint, filling_strategy, kwargs=kwargs)

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

    return [report_dataset]

  def _build_pdf_in_secondary_image_capture(self,
                                            datasets: Iterable[Dataset],
                                            report_blueprint,
                                            filling_strategy,
                                            document_bytes,
                                            kwargs: Dict) -> List[Dataset]:
    try:
      import pdf2image
    except ImportError as E: #pragma ignore
      raise MissingOptionalDependency("Missing package pdf2latex", E)

    pivot = get_pivot(datasets)
    if pivot is None:
      raise MissingPivotDataset

    image_pages = pdf2image.convert_from_bytes(document_bytes)

    report_datasets = []

    for page_number, image_page in enumerate(image_pages):
      kwargs['__dicom_factory_image'] = image_page
      # Page number are not 0-index
      kwargs['__dicom_factory_page_number'] = page_number + 1

      report_datasets.append(
        self.build_instance(pivot, report_blueprint, filling_strategy, kwargs)
      )

    return report_datasets

  REPORT_HANDLER_FUNCTIONS = {
    SecondaryCaptureImageStorage : _build_pdf_in_secondary_image_capture,
    EncapsulatedPDFStorage : _build_pdf_in_encapsulated_document
  }
