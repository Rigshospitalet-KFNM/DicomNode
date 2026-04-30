"""This module concern itself with defining user input. In here there's a
number of classes which you should use to define your input for your process function.


"""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard Library
from abc import abstractmethod, ABC, ABCMeta
from dataclasses import dataclass
from datetime import date
from enum import Enum
from functools import reduce
from logging import Logger, getLogger
from pathlib import Path
from operator import add
from threading import Thread
from typing import Any, Dict,List, Optional, Iterable, Tuple, Type, Union

# Third party packages
from pydicom import Dataset
from pydicom.datadict import tag_for_keyword
from pydicom.uid import UID

# Dicomnode packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.data_structures.defaulting_dict import DefaultingDict

from dicomnode.data_structures.optional import OptionalPath
from dicomnode.data_structures.storage import get_storage_from_config, Storage
from dicomnode.dicom.dimse import Address, QueryLevels,\
  AssociationContextManager, create_query_ae
from dicomnode.dicom.lazy_dataset import LazyDataset
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured,\
  InvalidTreeNode, ContractViolation
from dicomnode.lib.io import load_dicom, save_dicom, Directory
from dicomnode.lib.validators import get_validator_for_value, Validator
from dicomnode.lib.utils import name
from dicomnode.config import DicomnodeConfig, config_from_raw
from dicomnode.server.grinders import Grinder, IdentityGrinder

# Baby here we go!
class AbstractInputMetaClass(ABCMeta):
  """# Behold my infinite job security.

  Jokes aside, this class just enable the or operation between
  AbstractInputTypes

  So the user can go:
    UserAbstractInput_1 or UserAbstractInput_2
  In their type declaration.
  """
  type_options: List

  def __or__(self: Union[Type['AbstractInput'], Type['AbstractInputProxy']], # type: ignore
             value: Union[Type['AbstractInput'], Type['AbstractInputProxy']]
             ) -> Type['AbstractInputProxy']:
    if issubclass(self, AbstractInputProxy) and issubclass(value, AbstractInputProxy):
      self.type_options += value.type_options
      return self
    elif issubclass(self, AbstractInputProxy):
      self.type_options.append(value)
      return self
    elif issubclass(value, AbstractInputProxy):
      value.type_options.insert(0, self)
      return value

    class ProxyClass(AbstractInputProxy):
      type_options = [self, value]

    return ProxyClass


class AbstractInput(metaclass=AbstractInputMetaClass):
  """Container for dicom sets fulfilling the validate image function.

    Args:
        options (Options, optional): Options for the abstract input. Used to add
                                     Additional arguments to the
                                     Defaults to Options().
    """
  # Private tags should be injected, rather than put into the input
  _private_tags: Dict[int, Tuple[str, str, str, str, str]] = {}

  required_tags: List[Union[int,str]] | List[int] = [0x00080018] # SOPInstanceUID
  """The list of tags that must be present in a dataset to be accepted
  into the input. Consider checking SOP_mapping.py for collections of Tags."""

  required_values: Dict[Union[int,str], Any] = {}
  "A Mapping of tags and associated values, doesn't work for values in sequences"

  enforce_single_series = False
  """Ensures that the input only contains a single series. The first accepted
  series determines the SeriesUID"""

  enforce_single_study_date = False
  """Ensures that the input only contains """

  image_grinder: Grinder = IdentityGrinder()
  """Grinder for converting stored dicom images
  into a data usable by the processing function"""


  # Dunder methods
  def __init__(self,
      config: DicomnodeConfig = config_from_raw(),
      node_path = OptionalPath()
    ):
    super().__init__()
    self.options = config
    self.node_path = node_path
    self.images = 0

    self._study_date: Optional[str] = None # This gets updated by the PatientNode
    """Date for study, used with enforce. String is in format YYYYMMDD"""
    # There's something to be said about keeping it as string, because it's
    # objectively wrong that it's not a date object, but it's what dicom uses

    self.single_series_uid: Optional[UID] = None
    self.container: Optional[Directory] = Directory(node_path.path) if node_path else None
    self.logger = getLogger(DICOMNODE_LOGGER_NAME)

    # Tag for SOPInstance is (0x0008,0018)
    if 0x0008_0018 not in self.required_tags and "SOPInstanceUID" not in self.required_tags:
      self.logger.info(f"You should add SOPInstanceUID to required tags for {name(self)}")
      self.required_tags.append(0x0008_0018)

    self._storage = self._get_storage_type(config)(node_path / self.__class__.__name__)

  def __contains__(self, dataset: Dataset):
    return dataset in self._storage

  def __iter__(self):
    yield from self._storage

  def __len__(self):
    return len(self._storage)

  # abstract methods
  @abstractmethod
  def validate(self) -> bool:
    """Method for checking if all data needed for
processing is stored in the input. Should return `True` when input is ready for
processing, `False` otherwise.

    Returns:
        bool: If there's sufficient data to start processing
    """
    raise NotImplementedError #pragma: no cover

  def grind(self) -> Any:
    """This function retrieves all the data stores in the input,
    and makes it ready for processing

    Returns:
        Any: Data ready for the pipelines process function.
    """
    return self.image_grinder(self)

  def get_datasets(self) -> List[Dataset]:
    return [dataset for dataset in self]


  @classmethod
  def _validate_value(cls, value, target):
    return get_validator_for_value(value)(target)

  @classmethod
  def validate_image(cls, dicom: Dataset) -> bool:
    """Checks if an image belongs in the input

    This needs to be a classmethod because the proxy needs to use this method
    to determine if the proxy should instantiates this type

    Args:
        dicom (Dataset): Dataset in question

    Returns:
        bool: True if the image can be added, False if not
    """

    # Dataset Validation
    for required_tag in cls.required_tags:
      if isinstance(required_tag, str):
        required_tag = tag_for_keyword(required_tag)
        if required_tag is None:
          raise IncorrectlyConfigured(f"{required_tag} is not evaluate to a Dicom tag")

      if required_tag not in dicom:
        return False

    for required_tag, required_value in cls.required_values.items():
      if isinstance(required_tag, str):
        required_tag = tag_for_keyword(required_tag)
        if required_tag is None:
          raise IncorrectlyConfigured(f"{required_tag} is not evaluate to a Dicom tag")

      if required_tag not in dicom:
        return False

      if not cls._validate_value(required_value,dicom[required_tag].value):
        return False

    return True

  def _enforce_single_series(self, dicom):
    if self.enforce_single_series:
      if 0x0020_000E not in dicom: # 0x0020_000E = Series Instance UID
        return False
      if self.single_series_uid is None:
        self.single_series_uid = dicom[0x0020_000E].value
      else:
        if self.single_series_uid != dicom[0x0020_000E].value:
          return False
    return True

  def _enforce_date_requirement(self, dicom: Dataset) -> bool:
    """A state based check if the dicom is of the correct study date

    Args:
        dicom (Dataset): Dataset which must comply with date requirement

    Returns:
        bool: Returns False if dataset is of a different study date than this object
    """


    if self.enforce_single_study_date and self.study_date is not None:
      if 0x0008_0020 not in dicom or self.study_date != dicom.StudyDate: # 0x0008_0020 = Study Date
        return False
    return True

  def _get_storage_type(self, config: DicomnodeConfig):
    return get_storage_from_config(config)

  def _state_based_validation(self, dicom: Dataset) -> bool:
    """Check if a dicom can be accepted based on the state of this object

    Args:
        dicom (Dataset): Dataset to be checked

    Returns:
        bool: True if the dataset passes all checks, False if it fails one or more
    """
    return self._enforce_single_series(dicom) and self._enforce_date_requirement(dicom)

  def add_image(self, dicom: Dataset) -> int:
    """Attempts to add an image to the input.

    Args:
        dicom (Dataset): The dataset to be added

    Returns:
      int - The number of images added, in this case 1.

    Raises:
        InvalidDataset: If the dataset is not valid.
    """
    if not self.validate_image(dicom):
      raise InvalidDataset

    if not self._state_based_validation(dicom):
      raise InvalidDataset

    replaced = dicom in self
    # Save the dataset
    self._storage.store_image(dicom)

    if not replaced:
      self.images += 1
    return 1

  # This is a property because we need to overload the setter in historic input

  @property
  def study_date(self):
    """The study date - Note that it's the Owning patient Node that sets this!"""
    return self._study_date


  @study_date.setter
  def study_date(self, value):
    self._study_date = value

  @property
  def storage(self):
    """Gets the underlying storage container that actually holds all the images"""
    return self._storage

  def __str__(self):
    return f"{self.__class__.__name__} - {self.images} images - Valid: {self.validate()}"

class DynamicInput(AbstractInput):
  """This input signifies when you are dealing with a variable number of input series.

  Applies image grinder to each Input
  """
  separator_tag: int = 0x0020000E # SeriesInstanceUID

  def get_key(self, dataset) -> str:
    key = dataset[self.separator_tag].value
    if isinstance(key, UID):
      key = key.name
    if not isinstance(key, str):
      key = str(key)

    return key

  def __init__(self, config: DicomnodeConfig = config_from_raw(), node_path=OptionalPath()):
    super().__init__(config, node_path)

    def create_leaf(key: str):
      return self._get_storage_type(config)(node_path / key)
    self.leafs: DefaultingDict[str, Storage] = DefaultingDict(create_leaf)

  def __contains__(self, dataset: Dataset):
    key = self.get_key(dataset)

    return dataset in self.leafs[key]

  def __len__(self):
    return reduce(add, [len(storage) for key, storage in self.leafs], 0)

  def __iter__(self):
    for key, storage in self.leafs:
      yield from storage

  def grind(self) -> Dict[str, Any]:
    return_dict = {}
    for key, leaf in self.leafs:
      return_dict[key] = self.image_grinder(leaf)

    return return_dict

  @property
  def storage(self):
    raise ContractViolation("You cannot take the storage of a dynamic output")

  def add_image(self, dicom: Dataset) -> int:
    if not self.validate_image(dicom):
      raise InvalidDataset

    if self.separator_tag not in dicom:
      raise InvalidDataset

    key = self.get_key(dicom)

    self.leafs[key].store_image(dicom)

    self.images += 1
    return 1


class HistoricAbstractInput(AbstractInput):
  """This Input retrieves historic datasets based on the first dataset that this
  input accepts.

  As there quite a few pitfalls with this Input, Please read:
  https://dicomnode.readthedocs.io/en/latest/tutorials/configuring_a_pipeline.html#historic-inputs
  for proper configuration.

  """

  class HistoricGrinder(Grinder):
    def __call__(self, image_generator: Iterable[Dataset]) -> Any:
      if not isinstance(image_generator, HistoricAbstractInput):
        raise TypeError("The HistoricGrinder requires a HistoricAbstractInput")
      return image_generator.historic_dataset

  class HistoricInputState(Enum):
    EMPTY = 1
    FETCHING = 2
    FILLED = 3

  class HistoricAction(Enum):
    FIND_QUERY = 1
    MOVE_QUERY = 2

  address: Optional[Address] = None
  image_grinder = HistoricGrinder()

  def __init__(self, options: DicomnodeConfig, node_path= OptionalPath()):
    super().__init__(options, node_path)
    self.triggering_dataset = None

    if self.options.AE_TITLE is None:
      raise IncorrectlyConfigured(f"The historic Input: {name(self)} have been not parsed an AE title. This is violation from containing PatientNode!")
    self.ae_title = self.options.AE_TITLE

    if self.address is None or not isinstance(self.address, Address):
      raise IncorrectlyConfigured(f"{name(self)} needs an address to send images to!")
    self._address = self.address # This reassignment is really just for the linter, it assumes that

    if self.enforce_single_study_date:
      self.logger.warning(f"In {name(self)} enforce_single_study_date have been set, which is redundant for a historic input")

    self.historic_dataset: Dict[str, Dict[str, List[Dataset]]] = {}
    self.state = HistoricAbstractInput.HistoricInputState.EMPTY
    self.thread: Optional[Thread] = None

  def _enforce_date_requirement(self, dicom: Dataset):
    if self.study_date is None: # If study date have not been set yet
      return False

    if 'StudyDate' not in dicom:
      return False

    if self.study_date < dicom.StudyDate:
      return False

    return True


  @abstractmethod
  def check_query_dataset(self, current_study: Dataset, query_dataset: Optional[Dataset]=None) -> Optional[Tuple[HistoricAction,Dataset]]:
    return None

  def thread_target(self, query_data):
    query_level = QueryLevels(query_data.QueryRetrieveLevel)

    query_datasets = [query_data]

    with AssociationContextManager(
      create_query_ae(self.ae_title),
      self._address.ip,
      self._address.port,
      ae_title=self._address.ae_title
    ) as assoc:
      if assoc is None:
        self.logger.error(f"{name(self)} could connect to {self.ae_title} : ({self._address.ip},{self._address.port})")
        return

      studies_to_be_moved: List[Dataset] = []

      while len(query_datasets): # While there's dataset to query
        new_finds = []

        for query_dataset in query_datasets:
          find_response = assoc.send_c_find(query_dataset, query_level.find_sop_class())

          for (status, incoming_dataset) in find_response:
            if status.Status not in [0xFF00, 0x0000]:
              self.logger.error(f"While C-FIND'ing {name(self)} encountered a problem: {status}")

            if incoming_dataset is None:
              continue

            incoming_check = self.check_query_dataset(incoming_dataset, query_dataset)

            if incoming_check is not None:
              action, new_dataset = incoming_check

              if action == self.HistoricAction.FIND_QUERY:
                new_finds.append(new_dataset)
              elif action == self.HistoricAction.MOVE_QUERY:
                studies_to_be_moved.append(new_dataset)

        query_datasets = new_finds


      self.logger.info(f"{name(self)} found {len(studies_to_be_moved)} series to query for!")

      for study in studies_to_be_moved:
        move_responses = assoc.send_c_move(study, self.ae_title, query_level.move_sop_class())
        for status, move_response in move_responses:
          if status.Status not in [0xFF00, 0x0000]:
            self.logger.error(f"While C-Move'ing {name(self)} encountered a problem: {status}")
      # The datasets will then be added in add image and should be contained by
      # the HistoricINput

      # Note that you have set the state to filled while the association is live
      # otherwise if the historic association is the last to leave, there'll be
      # no thread to start the processing.
      self.logger.info(f"Historic input finished fetching history, and it now contains {len(self)} images")
      self.state = HistoricAbstractInput.HistoricInputState.FILLED

  def add_image(self, dicom: Dataset) -> int:
    if self.state == HistoricAbstractInput.HistoricInputState.EMPTY and 'StudyDate' in dicom:
      response = self.check_query_dataset(dicom)
      if response is not None:
        self.logger.debug("Historic input is now Fetching!")
        self.triggering_dataset = dicom
        action, query_dataset = response
        self.state = HistoricAbstractInput.HistoricInputState.FETCHING
        self.thread = Thread(group=None, name="Historic Input", target=self.thread_target, args=(query_dataset,))
        self.thread.start()
      return 0

    if not self._state_based_validation(dicom):
      #self.logger.debug(f"Historic Input rejecting dataset: {dicom.StudyDate} based on state: {self.study_date}")
      raise InvalidDataset


    if not self.validate_image(dicom):
      #self.logger.debug("Historic Input rejecting dataset based on static")
      raise InvalidDataset

    # This is mostly done to keep it simple
    #self.logger.debug("Historic input storing dataset")
    self._store_historic_dataset(dicom)

    return 1

  def _store_historic_dataset(self, historic_dataset: Dataset):
    if 'StudyDate' not in historic_dataset or 'SeriesDescription' not in historic_dataset:
      raise InvalidDataset

    study_date: str = historic_dataset.StudyDate # this might be a data-element
    if study_date not in self.historic_dataset:
      self.historic_dataset[study_date] = {}

    study_date_series = self.historic_dataset[study_date]

    series_description: str = historic_dataset.SeriesDescription
    if series_description not in study_date_series:
      study_date_series[series_description] = []

    datasets = study_date_series[series_description]
    datasets.append(historic_dataset)
    self.images += 1

  def _state_based_validation(self, dicom: Dataset) -> bool:
    if self.study_date is None:
      return False

    if 'StudyDate' not in dicom:
      return False

    return dicom.StudyDate < self.study_date

  def validate(self) -> bool:
    if self.thread is not None and self.state != HistoricAbstractInput.HistoricInputState.FILLED:
      self.thread.join(1.0)

    return self.state == HistoricAbstractInput.HistoricInputState.FILLED

  def __str__(self):
    study_info_strings = []

    for study_date, series_dict in self.historic_dataset.items():
      series_strings = []
      for series_name, list_of_datasets in series_dict.items():
        series_strings.append(f"    {series_name} - {len(list_of_datasets)} Datasets")
      series_string = "\n".join(series_strings)

      study_info_strings.append(f"  {study_date}:\n{series_string}")

    study_info_string = "\n".join(study_info_strings)
    return f"Historic Input - {self.state}\n{study_info_string}"

class AbstractInputProxy(AbstractInput):
  """Internal library Class, that is constructed from an or operation between
  two inputs. Despite using the or operator, it's actually a xor operation.

  Note that the operation create a superclass of this class, that then gets
  instantiated.

  Note that once proxy can determine what input it is, it evolves into the
  input

  """
  type_options: List[Type[AbstractInput]] # Set by MetaClass

  @property
  def images(self):
    return 0

  def __len__(self):
    return 0


  def validate(self) -> bool:
      return False

  def __init__(self, config: DicomnodeConfig = config_from_raw(), node_path = OptionalPath()):
    self.input_options = config
    self.node_path = node_path
    self._study_date = None
    enforcing_single_study_date = None
    for type_option in self.type_options:
      if not issubclass(type_option, AbstractInput):
        raise IncorrectlyConfigured(f"type option {type_option.__name__} is not a AbstractInputs")

      if enforcing_single_study_date is None:
        enforcing_single_study_date = type_option.enforce_single_study_date
      elif type_option.enforce_single_study_date != enforcing_single_study_date:
        raise IncorrectlyConfigured("You cannot create a union between non-enforcing study date and enforcing")

    if enforcing_single_study_date is None:
      raise IncorrectlyConfigured("A Proxy cannot proxy no object")
    else:
      self.enforce_single_study_date = enforcing_single_study_date

  def add_image(self, dicom: Dataset) -> int:
    if self.enforce_single_study_date:
      if not self._enforce_date_requirement(dicom):
        raise InvalidDataset
    for type_option in self.type_options:
      if type_option.validate_image(dicom):
        self.__class__ = type_option
        type_option.__init__(self, config=self.input_options, node_path=self.node_path)
        return self.add_image(dicom)
    raise InvalidDataset

__all__ = [
  'AbstractInput',
  'DynamicInput',
  'HistoricAbstractInput'
]
