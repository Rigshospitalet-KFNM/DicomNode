import logging
from typing import Any, Dict, List, Iterable, Tuple

from abc import ABC, abstractmethod

from pydicom import Dataset
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.lib.dimse import Address, send_images



logger = logging.getLogger("dicomnode")

class PipelineOutput(ABC):
  """Base Class for pipeline outputs.
  This class carries the responsibility for sending processed data to the endpoint

  """

  output: List[Tuple[Address, Any]]

  def __init__(self, output: List[Tuple[Address, Any]]) -> None:
    self.output = output

  @abstractmethod
  def send(self) -> bool:
    """Method that should send all the data in output

    Returns:
        bool: _description_
    """
    raise NotImplementedError

class DicomOutput(PipelineOutput):
  def __init__(self, output: List[Tuple[Address, Iterable[Dataset]]], AE: str) -> None:
    self.output = output
    self.ae = AE

  def send(self) -> bool:
    success = True
    for address, datasets in self.output:
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