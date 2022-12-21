import logging
from typing import Dict, List, Iterable, Tuple

from pydicom import Dataset

from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.lib.dimse import Address, send_images



logger = logging.getLogger("dicomnode")

class PipelineOutput:
  output: List[Tuple[Address, Iterable[Dataset]]]

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
  def __init__(self, output: List[Tuple[Address, Iterable[Dataset]]] = [], AE: str = "") -> None:
    super().__init__(output, AE)

  def send(self) -> bool:
    return True