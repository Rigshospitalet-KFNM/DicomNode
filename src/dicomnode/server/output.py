import logging

from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple, Type
from abc import ABC, abstractmethod

from pydicom import Dataset
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.lib.dimse import Address, send_images
from dicomnode.lib.imageTree import DicomTree, ImageTreeInterface


logger = logging.getLogger("dicomnode")

class PipelineOutput(ABC):
  """Base Class for pipeline outputs.
  This class carries the responsibility for sending processed data to the endpoint

  Should have an output attribute with the form
    Iterable[destination, payload]

  """

  output: List[Tuple[Any, Any]]

  def __init__(self, output: List[Tuple[Any, Any]]) -> None:
    self.output = output

  @abstractmethod
  def send(self) -> bool:
    """Method that should send all the data in output

    Returns:
        bool: _description_
    """
    raise NotImplementedError

  def __iter__(self):
    for destination, payload in self.output:
      yield destination, payload

class DicomOutput(PipelineOutput):
  def __init__(self, output: List[Tuple[Address, Iterable[Dataset]]], AE: str) -> None:
    self.output = output
    self.ae = AE

  def send(self) -> bool:
    success = True
    for address, datasets in self:
      try:
        send_images(self.ae, address, datasets)
      except CouldNotCompleteDIMSEMessage:
        logger.error(f"Could not send to images to {address.ae_title}")
        success = False
    return success

class NoOutput(PipelineOutput):
  output = []
  def __init__(self) -> None:
    pass

  def send(self) -> bool:
    return True

class FileOutput(PipelineOutput):
  image_tree_interface_type: Type[ImageTreeInterface]

  def __init__(self, output: List[Tuple[Path, Iterable[Dataset]]], image_tree_interface_type: Type[ImageTreeInterface]=DicomTree) -> None:
    self.output = output
    self.image_tree_interface_type = image_tree_interface_type

  def send(self) -> bool:
    for Path, Datasets in self.output:
      if not isinstance(Datasets, ImageTreeInterface):
        Datasets = self.image_tree_interface_type(Datasets)
      Datasets.save_tree(Path)
    return True
