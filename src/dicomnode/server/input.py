"""This module concern itself with defining user input. In here there's a
number of classes which you should use to define your input for your process function.


"""

from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict
from logging import Logger
from pathlib import Path
from pydicom import Dataset
from pydicom.uid import UID
from typing import List, Callable, Dict, Tuple, Any, Optional, Iterator, Type, Iterable, Union
import logging

from dicomnode.lib.dimse import Address, send_move_thread
from dicomnode.lib.dicomFactory import DicomFactory, Blueprint
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured, InvalidTreeNode
from dicomnode.lib.io import load_dicom, save_dicom
from dicomnode.lib.lazyDataset import LazyDataset
from dicomnode.lib.grinders import identity_grinder, dicom_tree_grinder
from dicomnode.lib.imageTree import ImageTreeInterface
from dicomnode.lib.utils import staticfy


class AbstractInput(ImageTreeInterface, ABC):
  required_tags: List[int] = [0x00080018, 0x7FE00010] # InstanceUID, Pixel Data
  __private_tags: Dict[int, Tuple[str, str, str, str, str]] = {}
  required_values: Dict[int, Any] = {}
  image_grinder: Callable[[Iterator[Dataset]], Any] = identity_grinder

  @dataclass
  class Options: # These are options that are injected into all input.
    # Note the reason, why there some options, that are not used by this class
    # is because of Liskov's Substitution principle, and subclasses might need
    # these options.
    ae_title: Optional[str] = None
    logger: Optional[Logger] = None
    data_directory: Optional[Path]  = None
    factory: Optional[DicomFactory] = None
    lazy: bool = False

  def __init__(self,
      pivot: Optional[Dataset] = None,
      options: Options = Options(),
    ):
    super().__init__()
    self.options = options

    self.path: Optional[Path] = options.data_directory
    if self.options.logger is not None:
      self.logger = self.options.logger
    else:
      self.logger = logging.getLogger("dicomnode")

    if 0x00080018 not in self.required_tags: # Tag for SOPInstance is (0x0008,0018)
      self.required_tags.append(0x00080018)

    if self.path is not None:
      if not self.path.exists():
        self.path.mkdir(exist_ok=True)
      for image_path in self.path.iterdir():
        dcm = load_dicom(image_path, self.__private_tags)
        self.add_image(dcm)

  @abstractmethod
  def validate(self) -> bool:
    """Checks if the input have sufficient data, to start processing

    Returns:
        bool: If there's sufficient data to start processing
    """
    raise NotImplementedError #pragma: no cover

  def _clean_up(self) -> None:
    """Removes any files, stored by the Input"""
    if self.path is not None:
      for dicom in self:
        p = self.get_path(dicom)
        p.unlink()

  def get_data(self) -> Any:
    """This function retrieves all the data stores in the input,
    and makes it ready for processing

    Returns:
        Any: Data ready for the pipelines process function.
    """
    return staticfy(self.image_grinder)(self)

  def get_path(self, dicom: Dataset) -> Path:
    """Gets the path, where a dataset would be saved.

    Args:
        dicom (Dataset): The dataset in question

    Returns:
        Path: The path for that dataset.

    Raises:
      IncorrectlyConfigured : Calls to this function require a dictory
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
        self.logger.debug(f"required tag: {hex(required_tag)} in dicom")
        return False

    for required_tag, required_value in self.required_values.items():
      if required_tag not in dicom:
        self.logger.debug(f"required value tag: {hex(required_tag)} in dicom")
        return False
      if dicom[required_tag].value != required_value:
        self.logger.debug(f"required value {required_value} not match {dicom[required_tag]} in dicom")
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
  def __init__(self, dcm: Union[Iterable[Dataset], Dataset] = [], lazy = False, path: Optional[Path] = None) -> None:
    super().__init__(dcm)
    self.lazy = lazy
    self.path = path

  def get_path(self, dicom: Dataset) -> Path:
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
    returnDict = {}
    for key, leaf in self.data.items():
      if not isinstance(leaf, DynamicLeaf):
        raise InvalidTreeNode # pragma: no cover

      returnDict[key] = staticfy(self.image_grinder)(list(leaf.data.values()))

    return returnDict

  def add_image(self, dataset: Dataset) -> int:
    if not self.validate_image(dataset):
      raise InvalidDataset

    if self.separator_tag not in dataset:
      raise InvalidDataset

    key = dataset[self.separator_tag].value
    if isinstance(key, UID):
      key = key.name
      # This is to ensure the assumption that underlying data dict is Dict[str, Union[Dataset, ImageTreeInterface]]
    if not isinstance(key, str):
      key = str(key) # Otherwise the imageTree throws a type error

    if key in self:
      image_tree = self[key]
      if isinstance(image_tree, ImageTreeInterface):
        ret_value = image_tree.add_image(dataset)
      else:
        raise InvalidTreeNode #pragma: no cover
    else:
      # Don't use the add image functionality of the constructor due to fact that, it's return value is needed
      if self.path is not None:
        leaf_path = self.path / key
        leaf_path.mkdir(parents=True, exist_ok=True)
      else:
        leaf_path = None
      leaf = self.leaf_class([], self.options.lazy, leaf_path)
      self[key] = leaf
      ret_value = leaf.add_image(dataset)
    self.images += ret_value
    return ret_value


class HistoricAbstractInput(AbstractInput):
  address: Optional[Address] = None
  c_move_blueprint: Optional[Blueprint] = None

  def __init__(self, pivot: Optional[Dataset] = None, options: AbstractInput.Options = AbstractInput.Options()):
    super().__init__(pivot, options)

    if pivot is None:
      self.logger.critical("You forgot to parse the pivot to The Input")
      raise IncorrectlyConfigured

    if self.c_move_blueprint is None:
      self.logger.critical("A C move blueprint is missing")
      raise IncorrectlyConfigured

    if self.address is None:
      self.logger.critical("A target address is needed to send a C-Move to")
      raise IncorrectlyConfigured

    if self.options.factory is None:
      self.logger.critical("A Factory is needed to generate a C move message")
      raise IncorrectlyConfigured

    if self.options.ae_title is None:
      self.logger.critical("Historic Inputs needs a AE Title of the SCU")
      raise IncorrectlyConfigured

    message = self.options.factory.build(pivot,self.c_move_blueprint)

    send_move_thread(self.options.ae_title, self.address, message)

