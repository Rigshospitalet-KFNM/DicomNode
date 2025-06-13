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
    Optional,Sequence as TypingSequence, Tuple, TypeVar, Union, TypeAlias

# Third Party Library
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from numpy import ndarray, zeros_like
from pydicom import DataElement, Dataset, Sequence, datadict
from pydicom.uid import EncapsulatedPDFStorage, SecondaryCaptureImageStorage
from pydicom.tag import Tag, BaseTag
from sortedcontainers import SortedDict

# Dicomnode Library
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING, SIGNED_ARRAY_ENCODING, DICOMNODE_PRIVATE_TAG_VERSION, DICOMNODE_PRIVATE_TAG_HEADER
from dicomnode.dicom import make_meta, Reserved_Tags
from dicomnode.dicom.series import DicomSeries, NiftiSeries
from dicomnode.math.image import fit_image_into_unsigned_bit_range
from dicomnode.lib.exceptions import IncorrectlyConfigured, InvalidDataset, MissingOptionalDependency
from dicomnode.lib.logging import get_logger
from dicomnode.lib.exceptions import MissingPivotDataset, ConstructionFailure

logger = get_logger()

T = TypeVar('T')

TagType: TypeAlias = Union[BaseTag, str, int, Tuple[int,int]]

def get_pivot(datasets: Union[Dataset,Iterable[Dataset]]) -> Optional[Dataset]:
  if isinstance(datasets, Dataset):
    return datasets
  dataset = None
  for _dataset in datasets:
    dataset = _dataset
    break
  return dataset


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
  def tag(self, tag: TagType):
    self.__tag = Tag(tag)

  @property
  def VR(self) -> str:
    return self.__VR

  @property
  def is_private(self) -> bool:
    return self.tag.is_private

  @VR.setter
  def VR(self, val: str) -> None:
    self.__VR = val

  def __init__(self,
               tag: TagType,
               VR: str,
               name: Optional[str]=None) -> None:
    self.tag = tag
    self.VR = VR
    if name is None:
      self.name = DataElement(tag, VR, None).name
    elif 64 < len(name):
      logger.warning(f"Virtual Element name is being truncated from {name} to {name[64:]}")
      self.name = name[64:]
    else:
      self.name = name

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

  def __init__(self, tag: TagType, Optional: bool = False, name: Optional[str] = None) -> None:
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

  def corporealialize(self, datasets: Iterable[Dataset]) -> None:
    return None


class StaticElement(VirtualElement, Generic[T]):
  def __init__(self,
               tag: Union[str, BaseTag, int, Tuple[int, int]],
               VR: str, value: T,
               name:Optional[str]=None) -> None:
    super().__init__(tag, VR, name=name)
    self.value: T = value

  def corporealialize(self, datasets: Iterable[Dataset]) -> DataElement:
    return DataElement(self.tag, self.VR, self.value)

  def __str__(self):
    return f"<StaticElement {self.name}: {self.tag} {self.VR} {self.value}>"

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
  image: Any = None

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
                      datasets: Iterable[Dataset]) -> 'SequenceElement':
    self._partial_initialized_sequences = []
    for blueprint in self._blueprints:
      partial_initialized_sequence = []
      for virtual_element in blueprint:
        corporealialize_value = virtual_element.corporealialize(datasets)
        if corporealialize_value is not None:
          partial_initialized_sequence.append(corporealialize_value)
      self._partial_initialized_sequences.append(partial_initialized_sequence)

    return self

  def produce(self, instance_environment: InstanceEnvironment) -> DataElement:
    if self._partial_initialized_sequences is None:
      logger.error("You are attempting to produce from an uninitialized sequence element")
      raise ConstructionFailure(f"You are attempting to produce from an uninitialized sequence element at tag {self.tag}")

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

class IndexElement(InstanceVirtualElement):
  def __init__(self, tag: TagType, VR: str, indexable: TypingSequence[DataElement], name: str | None = None) -> None:
    super().__init__(tag, VR, name)
    self.indexable = indexable

  def corporealialize(self, datasets: Iterable[Dataset]) -> DataElement | InstanceVirtualElement | None:
    return self

  def produce(self, instance_environment: InstanceEnvironment) -> DataElement:
    return self.indexable[instance_environment.instance_number]

class KeyedIndexElement(InstanceVirtualElement):
  def __init__(self, tag: TagType, VR: str, key: Any, name: str | None = None) -> None:
    super().__init__(tag, VR, name)
    self.key = key

  def corporealialize(self, datasets: Iterable[Dataset]) -> DataElement | InstanceVirtualElement | None:
    return self

  def produce(self, instance_environment: InstanceEnvironment) -> DataElement:
    return instance_environment.kwargs[self.key][instance_environment.instance_number]


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

  def produce(self, instance_environment: InstanceEnvironment) -> DataElement:
    return DataElement(self.tag, self.VR, self.func(instance_environment))


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
      raise ConstructionFailure(f"Unable to construct {self.tag} because there was no reference dataset")

    return instance_environment.instance_dataset.get(self.tag,
                                                     DataElement(self.tag, self.VR, None))

class CopyOrElseElement(InstanceVirtualElement):
  """Like the Copy Element, but adds the functionality to place a
  missing value instead of just nothing"""
  def __init__(self, tag: TagType, VR: str, orElse: Any, name=None):
    super().__init__(tag, VR, name)
    self._orElse = orElse

  def corporealialize(self, datasets: Iterable[Dataset]) -> DataElement | InstanceVirtualElement | None:
    dataset = get_pivot(datasets)

    if dataset is not None:
      if self.tag in dataset:
        return dataset[self.tag]

    if isinstance(self._orElse, DataElement):
      return self._orElse
    elif isinstance(self._orElse, VirtualElement):
      return self._orElse.corporealialize(datasets)
    else:
      return DataElement(self.tag, self.VR, self._orElse)

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

  def __contains__(self, tag: int) -> bool:
    return tag in self._dict

  def __len__(self):
    return len(self._dict)

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
      tags = list(self._dict.irange(tag_group, tag))
      if tags:
        if tags[-1] < tag:
          index = len(tags)
        else:
          index = len(tags) -1
      else:
        index = 0
      tag_name = tag_group + Reserved_Tags.PRIVATE_TAG_NAMES.value
      tag_VR = tag_group + Reserved_Tags.PRIVATE_TAG_VRS.value

      # Check if the Group is allocated

      if not group_id in self:
        self._dict[group_id] = StaticElement[str](group_id, "LO", DICOMNODE_PRIVATE_TAG_HEADER)
        self._dict[tag_name] = StaticElement[List[str]](tag_name, "LO", [])
        self._dict[tag_VR] = StaticElement[List[str]](tag_VR, "LO", [])

      if tag in self:
        for reserved_tag_header in Reserved_Tags:
          static_element: StaticElement[List[str]] = self[tag_group + reserved_tag_header.value] # type: ignore
          del static_element.value[index]

      VE_names: StaticElement[List[str]] = self._dict[tag_name]
      VE_names.value.insert(index, f"({virtual_element.tag.group:04x}, {virtual_element.tag.element:04x}) - {virtual_element.name}")
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

    match self.default_pixel_representation:
      case self.PixelRepresentation.UNSIGNED:
        stored_image_type = UNSIGNED_ARRAY_ENCODING.get(self.default_bits_allocated, None)
        if stored_image_type is None:
          raise IncorrectlyConfigured("default bits allocated must be 8,16,32,64")
      case self.PixelRepresentation.TWOS_COMPLIMENT:
        raise NotImplementedError("Signed encoding is not supported yet")
      case _:
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
                   image,
                   blueprint: Blueprint,
                   parent_series: Union[DicomSeries, List[Dataset]],
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

    if isinstance(parent_series, List):
      parent_datasets = parent_series
    else:
      parent_datasets = parent_series.datasets

    can_copy_instances = len(parent_datasets) == len(image)

    if not can_copy_instances:
      logger.info(f"Constructing a series of length {len(image)}, but there's only supplied {len(parent_datasets)} datasets!")

    series = DicomSeries([Dataset() for _ in image])

    for virtual_tag in blueprint:
      try:
        result = virtual_tag.corporealialize(parent_datasets)
      except Exception as exception:
        logger.error(f"Encountered an error in corporalizing Tag:{virtual_tag.tag} - {datadict.keyword_for_tag(virtual_tag.tag)}!")
        raise exception
      if isinstance(result, DataElement):
        series.set_shared_tag(virtual_tag.tag, result)
      elif isinstance(result, InstanceVirtualElement):
        if can_copy_instances:
          envs = [
            InstanceEnvironment(
              instance_number=i + 1, # InstanceNumbers are 1 indexed
              factory=self,
              kwargs=kwargs,
              instance_dataset=dataset,
              image=slice_,
            ) for (i, (dataset, slice_)) in enumerate(zip(parent_datasets, image))
          ]
        else:
          envs = [
            InstanceEnvironment(
              instance_number=i,
              factory=self,
              kwargs=kwargs,
              image=slice_,
            ) for i, slice_ in enumerate(image)
          ]

        data_elements = [
          result.produce(env) for env in envs
        ]

        series.set_individual_tag(virtual_tag.tag,
                                  data_elements)

    for i,dataset in enumerate(series.datasets):
      self.store_image_in_dataset(dataset, image[i])
      make_meta(dataset)

    return series

  def build_nifti_series(self,
                         nifti: Union[Nifti1Image, Nifti2Image, NiftiSeries],
                         blueprint: Blueprint,
                         kwargs : Dict[Any,Any]
                         ) -> DicomSeries:
    """Builds a Dicom series from a nifti series and a blueprint
    Note that the following tags doesn't need to be included


    Args:
        nifti (Union[Nifti1Image, Nifti2Image, NiftiSeries]): _description_
        blueprint (Blueprint): _description_
        kwargs (Dict[Any,Any]): _description_

    Returns:
        DicomSeries: _description_

    Throws:
      ConstructionFailure : If a CopyVirtualElement is inside of the
                            blueprint. As there's no parent series to copy
                            from
    """
    if not isinstance(nifti, NiftiSeries):
      nifti = NiftiSeries(nifti)

    corporealialized_blueprint = [virtual_tag.corporealialize([]) for virtual_tag in blueprint]

    datasets = []
    if nifti.image.raw.ndim == 3:
      for i, slice_ in enumerate(nifti.image):
        ds = Dataset()

        for corporealialized_tag in corporealialized_blueprint:
          if isinstance(corporealialized_tag, DataElement):
            ds[corporealialized_tag.tag] = corporealialized_tag
          elif isinstance(corporealialized_tag, InstanceVirtualElement):
            instance_environment = InstanceEnvironment(i + 1, self, image=slice_, kwargs=kwargs)

            ds[corporealialized_tag.tag] = corporealialized_tag.produce(
              instance_environment
            )

        datasets.append(ds)

    return DicomSeries(datasets)

  def build_series_without_image_encoding(self,
                                          images: TypingSequence,
                                          blueprint: Blueprint,
                                          datasets: Union[DicomSeries, List[Dataset]],
                                          kwargs: Dict,
                                          ):
    if isinstance(datasets, List):
      parent_datasets = datasets
    else:
      parent_datasets = datasets.datasets

    can_copy_instances = len(parent_datasets) == len(images)
    build_series = DicomSeries([Dataset() for _ in images])

    for virtual_element in blueprint:
      de = virtual_element.corporealialize(parent_datasets)
      if isinstance(de, DataElement):
        build_series.set_shared_tag(virtual_element.tag, de)
      elif isinstance(de, InstanceVirtualElement):
        if can_copy_instances:
          envs = [
            InstanceEnvironment(
              instance_number=i + 1,
              factory=self,
              kwargs=kwargs,
              instance_dataset=dataset,
              image=slice_,
            ) for (i, (dataset, slice_)) in enumerate(zip(parent_datasets, images))
          ]
        else:
          envs = [
            InstanceEnvironment(
              instance_number=i + 1,
              factory=self,
              kwargs=kwargs,
              image=slice_,
            ) for i, slice_ in enumerate(images)
          ]

        data_elements = [
          de.produce(env) for env in envs
        ]

        build_series.set_individual_tag(virtual_element.tag,
                                        data_elements)

    for dataset in build_series.datasets:
      make_meta(dataset)

    return build_series


  def build_instance(self,
            pivot: Dataset,
            blueprint: Blueprint,
            kwargs: Dict[Any, Any] = {}) -> Dataset:
    """Builds a singular dataset from blueprint and pivot dataset

    Intended to be used to singular such as messages or report datasets

    Args:
        pivot (Dataset): Dataset the blueprint will use to extract data from
        blueprint (Blueprint): Determines what data will be in the newly
                               constructed dataset

    Returns:
        Dataset: The constructed dataset
    """

    dataset = Dataset()
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
                 datasets: List[Dataset],
                 report_blueprint: Blueprint,
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
      return handler_function(self, datasets,
                              report_blueprint,
                              document_bytes,
                              kwargs)
    raise ValueError(f"No handler function for {report_class_UID}")

  def _build_pdf_in_encapsulated_document(self,
                                          datasets,
                                          report_blueprint,
                                          document_bytes,
                                          kwargs) -> List[Dataset]:
    pivot = get_pivot(datasets)
    if pivot is None:
      raise MissingPivotDataset
    report_dataset = self.build_instance(pivot, report_blueprint, kwargs=kwargs)

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
                                            datasets: List[Dataset],
                                            report_blueprint : Blueprint,
                                            document_bytes,
                                            kwargs: Dict) -> List[Dataset]:
    try:
      import pdf2image
    except ImportError as E: #pragma: ignore
      raise MissingOptionalDependency("Missing package pdf2image", E)

    pivot = get_pivot(datasets)
    if pivot is None:
      raise MissingPivotDataset

    image_pages = pdf2image.convert_from_bytes(document_bytes)

    series = self.build_series_without_image_encoding(image_pages, report_blueprint, datasets, kwargs)

    return series.datasets

  REPORT_HANDLER_FUNCTIONS = {
    SecondaryCaptureImageStorage : _build_pdf_in_secondary_image_capture,
    EncapsulatedPDFStorage : _build_pdf_in_encapsulated_document
  }
