"""This module concern itself with defining user input. In here there's a
number of classes which you should use to define your input for your process function.


"""

from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict
from logging import Logger
from pathlib import Path
from pydicom import Dataset
from typing import List, Callable, Dict, Tuple, Any, Optional, Iterator, Type
import logging

from dicomnode.lib.dicomFactory import DicomFactory
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured
from dicomnode.lib.io import load_dicom, save_dicom
from dicomnode.lib.grinders import identity_grinder
from dicomnode.lib.imageTree import ImageTreeInterface
from dicomnode.lib.utils import staticfy


class AbstractInput(ImageTreeInterface, ABC):
  required_tags: List[int] = [0x00080018, 0x7FE00010] # InstanceUID, Pixel Data
  __private_tags: Dict[int, Tuple[str, str, str, str, str]] = {}
  required_values: Dict[int, Any] = {}
  image_grinder: Callable[[Iterator[Dataset]], Any] = identity_grinder

  @dataclass
  class Options:
    logger: Optional[Logger] = None
    data_directory: Optional[Path]  = None
    factory: Optional[DicomFactory] = None

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
        p = self.__getPath(dicom)
        p.unlink()

  def get_data(self) -> Any:
    """This function retrieves all the data stores in the input,
    and makes it ready for processing

    Returns:
        Any: Data ready for the pipelines process function.
    """
    return staticfy(self.image_grinder)(self)

  def __getPath(self, dicom: Dataset) -> Path:
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

  def add_image(self, dicom: Dataset) -> int:
    """Attempts to add an image to the input.

    Args:
        dicom (Dataset): The dataset to be added

    Returns:
      int - The number of images added, in this case 1.

    Raises:
        InvalidDataset: If the dataset is not valid.
    """
    # Dataset Validation
    for required_tag in self.required_tags:
      if required_tag not in dicom:
        self.logger.debug(f"required tag: {hex(required_tag)} in dicom")
        raise InvalidDataset()

    for required_tag, required_value in self.required_values.items():
      if required_tag not in dicom:
        self.logger.debug(f"required value tag: {hex(required_tag)} in dicom")
        raise InvalidDataset()
      if dicom[required_tag].value != required_value:
        self.logger.debug(f"required value {required_value} not match {dicom[required_tag]} in dicom")
        raise InvalidDataset()

    # Save the dataset
    self[dicom.SOPInstanceUID.name] = dicom # Tag for SOPInstance is (0x0008,0018)
    self.images += 1
    if self.path is not None:
      dicom_path:Path = self.__getPath(dicom)
      if not dicom_path.exists():
        save_dicom(dicom_path, dicom)
    return 1

  def __str__(self) -> str:
    return str(self.data)

