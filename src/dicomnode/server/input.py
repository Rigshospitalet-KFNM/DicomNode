"""This module concern itself with defining user input. In here there's a
number of classes which you should use to define your input for your process function.


"""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard Library
from abc import abstractmethod, ABC
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from re import Pattern
from typing import List, Dict, Tuple, Any, Optional, Type, Iterable, Union

# Third party packages
from pydicom import Dataset
from pydicom.uid import UID

# Dicomnode packages
from dicomnode.data_structures.image_tree import ImageTreeInterface
from dicomnode.dicom.dimse import Address, send_move_thread, QueryLevels
from dicomnode.dicom.dicom_factory import DicomFactory, Blueprint
from dicomnode.dicom.lazy_dataset import LazyDataset
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured, InvalidTreeNode
from dicomnode.lib.io import load_dicom, save_dicom
from dicomnode.lib.logging import get_logger
from dicomnode.server.grinders import Grinder, IdentityGrinder

class AbstractInput(ImageTreeInterface, ABC):
  """Container for dicom sets fulfilling the validate image function.

    Args:
        options (Options, optional): Options for the abstract input. Used to add
                                     Additional arguments to the
                                     Defaults to Options().
    """
  # Private tags should be injected, rather than put into the input
  _private_tags: Dict[int, Tuple[str, str, str, str, str]] = {}

  required_tags: List[int] = [0x00080018] # SOPInstanceUID
  """The list of tags that must be present in a dataset to be accepted
  into the input. Consider checking SOP_mapping.py for collections of Tags."""

  required_values: Dict[int, Any] = {}
  "A Mapping of tags and associated values, doesn't work for values in sequences"

  image_grinder: Grinder = IdentityGrinder()
  """Grinder for converting stored dicom images
  into a data usable by the processing function"""

  @dataclass
  class Options:
    """These are the options to an abstract input"""
    # These are options that are injected into all input.
    # Note the reason, why there some options, that are not used by this class
    # is because of Liskov's Substitution principle, and subclasses might need
    # these options.
    ae_title: Optional[str] = None
    logger: Optional[Logger] = None
    data_directory: Optional[Path]  = None
    lazy: bool = False
    "Indicate if the Abstract input should use "

  def __init__(self,
      options: Options = Options(),
    ):
    super().__init__()
    self.options = options
    "Options for this Abstract input"

    self.path: Optional[Path] = options.data_directory
    if self.options.logger is not None:
      self.logger = self.options.logger
      "Logger for logging"
    else:
      self.logger = get_logger()

    # Tag for SOPInstance is (0x0008,0018)
    if 0x00080018 not in self.required_tags:
      self.required_tags.append(0x00080018)

    if self.path is not None:
      if not self.path.exists():
        self.path.mkdir(exist_ok=True)
      for image_path in self.path.iterdir():
        dcm = load_dicom(image_path)
        self.add_image(dcm)


  @abstractmethod
  def validate(self) -> bool:
    """Checks if the input have sufficient data, to start processing

    Returns:
        bool: If there's sufficient data to start processing
    """
    raise NotImplementedError #pragma: no cover

  def clean_up(self) -> int:
    """Removes any files, stored by the Input"""
    if self.path is not None:
      for dicom in self:
        path = self.get_path(dicom)
        path.unlink()
    return self.images

  def get_data(self) -> Any:
    """This function retrieves all the data stores in the input,
    and makes it ready for processing

    Returns:
        Any: Data ready for the pipelines process function.
    """
    return self.image_grinder(self)

  def get_datasets(self) -> List[Dataset]:
    return [dataset for dataset in self]

  def get_path(self, dicom: Dataset) -> Path:
    """Gets the path, where a dataset would be saved.

    Args:
        dicom (Dataset): The dataset in question

    Returns:
        Path: The path for that dataset.

    Raises:
      IncorrectlyConfigured : Calls to this function require a directory
    """
    if self.path is None:
      raise IncorrectlyConfigured

    image_name: str = ""
    if 0x00080060 in dicom: # Modality
      image_name += f"{dicom.Modality}_"
    image_name += "image"

    if 0x00200013 in dicom: # Instance Number
      image_name += f"_{dicom.InstanceNumber}"
    else:
      image_name += f"_{dicom.SOPInstanceUID.name}"

    image_name += ".dcm"

    return self.path / image_name

  def _validate_value(self, value, target):
    if isinstance(target, Pattern):
      return target.match(value) is not None
    return value == target

  def validate_image(self, dicom: Dataset) -> bool:
    """Checks if an image belongs in the input

    Args:
        dicom (Dataset): Dataset in question

    Returns:
        bool: True if the image can be added, False if not
    """
    # Dataset Validation
    for required_tag in self.required_tags:
      if required_tag not in dicom:
        #self.logger.debug(f"required tag: {hex(required_tag)} in dicom")
        return False

    for required_tag, required_value in self.required_values.items():
      if required_tag not in dicom:
        #self.logger.debug(f"required value tag: {hex(required_tag)} in dicom")
        return False
      if not self._validate_value(dicom[required_tag].value, required_value):
        #self.logger.debug(f"required value {required_value} not match {dicom[required_tag]} in dicom")
        return False

    return True

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

    # Save the dataset
    if self.options.lazy:
      if self.path is None:
        raise IncorrectlyConfigured("Lazy object require file storage")
      dicom_path = self.get_path(dicom)
      if not dicom_path.exists():
        save_dicom(dicom_path, dicom)
      self[dicom.SOPInstanceUID.name] = LazyDataset(dicom_path)
    else:
      self[dicom.SOPInstanceUID.name] = dicom # Tag for SOPInstance is (0x0008,0018)
      if self.path is not None:
        dicom_path = self.get_path(dicom)
        if not dicom_path.exists():
          save_dicom(dicom_path, dicom)
    self.images += 1
    return 1

class DynamicLeaf(ImageTreeInterface):
  """Subclass to DynamicInput, each instance is a separate series"""
  def __init__(self,
               dcm: Union[Iterable[Dataset], Dataset],
               lazy = False,
               path: Optional[Path] = None) -> None:
    super().__init__(dcm)
    self.lazy = lazy
    self.path = path

  def get_path(self, dicom: Dataset) -> Path:
    """Retrieves the path of a dataset, if it would be stored in this this node
    on the file system

    Args:
        dicom (Dataset): dataset to be stored

    Raises:
        IncorrectlyConfigured: If the DynamicLeaf is configured only to work in memory

    Returns:
        Path: Path where this dataset would be stored
    """
    if self.path is None:
      raise IncorrectlyConfigured("getting the path needs a base path")
    return self.path / (dicom.SOPInstanceUID.name + ".dcm")

  def add_image(self, dicom: Dataset) -> int:
    if self.lazy:
      if self.path is None:
        raise IncorrectlyConfigured("Lazy datasets require a path")
      dicom_path = self.get_path(dicom)
      self[dicom.SOPInstanceUID.name] = LazyDataset(dicom_path)
    else:
      self[dicom.SOPInstanceUID.name] = dicom # Tag for SOPInstance is (0x0008,0018)
      if self.path is not None:
        dicom_path = self.get_path(dicom)
        if not dicom_path.exists():
          save_dicom(dicom_path, dicom)
    self.images += 1
    return 1

class DynamicInput(AbstractInput):
  """This input signifies when you are dealing with a variable number of input series.

  Applies image grinder to each Input
  """
  leaf_class: Type[DynamicLeaf] = DynamicLeaf
  separator_tag: int = 0x0020000E # SeriesInstanceUID

  def get_data(self) -> Dict[str, Any]:
    return_dict = {}
    for key, leaf in self.data.items():
      if not isinstance(leaf, DynamicLeaf):
        raise InvalidTreeNode # pragma: no cover
      return_dict[key] = self.image_grinder(leaf)

    return return_dict

  def add_image(self, dicom: Dataset) -> int:
    if not self.validate_image(dicom):
      raise InvalidDataset

    if self.separator_tag not in dicom:
      raise InvalidDataset

    key = dicom[self.separator_tag].value
    if isinstance(key, UID):
      key = key.name
      # This is to ensure the assumption that underlying data dict is:
      #  Dict[str, Union[Dataset, ImageTreeInterface]]
    if not isinstance(key, str):
      key = str(key) # Otherwise the imageTree throws a type error

    if key in self:
      image_tree = self[key]
      if isinstance(image_tree, ImageTreeInterface):
        ret_value = image_tree.add_image(dicom)
      else:
        raise InvalidTreeNode #pragma: no cover
    else:
      # Don't use the add image functionality of the constructor due to fact
      # that, it's return value is needed
      if self.path is not None:
        leaf_path = self.path / key
        leaf_path.mkdir(parents=True, exist_ok=True)
      else:
        leaf_path = None
      leaf = self.leaf_class([], self.options.lazy, leaf_path)
      self[key] = leaf
      ret_value = leaf.add_image(dicom)
    self.images += ret_value
    return ret_value


class HistoricAbstractInput(AbstractInput, ABC):
  """This input sends a DIMSE C-MOVE to location on creation

  An Example could be that your node receives a pet image, then this input can
  fetch some CT stored at some database.

  Raises:
      IncorrectlyConfigured: _description_
      IncorrectlyConfigured: _description_
      IncorrectlyConfigured: _description_
      IncorrectlyConfigured: _description_
      IncorrectlyConfigured: _description_
  """

  address: Address = None
  query_level: QueryLevels = None

  def __init__(self, options: AbstractInput.Options = AbstractInput.Options()):
    super().__init__(options)
    self.send_historic_message = False # To do lock it
    if self.address is None or self.query_level is None:
      raise IncorrectlyConfigured("Historic datasets needs an Address and query level defined")

  @abstractmethod
  def get_message_dataset(added_dataset: Dataset) -> Dataset:
    raise NotImplemented # pragma: type ignore

  def add_image(self, dataset: Dataset) -> int:
    images = super().add_image(dataset)

    if not self.send_historic_message and 0 < images:
      self.send_historic_message = True
      message = self.get_message_dataset(dataset)
      send_move_thread(
        SCU_AE=self.options.ae_title,
        address=self.address,
        dataset=message,
      )

__all__ = [
  'AbstractInput',
  'DynamicInput',
  'DynamicLeaf',
  'HistoricAbstractInput'
]
