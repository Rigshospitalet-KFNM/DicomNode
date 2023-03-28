import logging

from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple, Type, Callable
from abc import ABC, abstractmethod

from pydicom import Dataset
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.lib.dimse import Address, send_images
from dicomnode.lib.image_tree import DicomTree, ImageTreeInterface
from dicomnode.lib.io import save_dicom


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
    raise NotImplementedError # pragma: no cover

  def __iter__(self):
    for destination, payload in self.output:
      yield destination, payload

class DicomOutput(PipelineOutput):
  output: List[Tuple[Address, Iterable[Dataset]]]
  def __init__(self, output: List[Tuple[Address, Iterable[Dataset]]], AE: str) -> None:
    self.ae = AE
    super().__init__(output)

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

def save_dataset(path, dataset) -> None:
  dataset_path: Path = path / dataset.StudyInstanceUID.name / dataset.SeriesInstanceUID.name / (dataset.SOPInstanceUID.name + '.dcm')
  save_dicom(dataset_path,  dataset)
  return None

class FileOutput(PipelineOutput):
  saving_function: Callable[[Path, Dataset], None]
  output: List[Tuple[Path, Iterable[Dataset]]]

  def __init__(self,
      output: List[Tuple[Path, Iterable[Dataset]]],
      saving_function: Callable[[Path, Dataset], None]=save_dataset) -> None:
    super().__init__(output)
    self.saving_function = saving_function

  def send(self) -> bool:
    for path, datasets in self.output:
      for dataset in datasets:
        self.saving_function(path, dataset)
    return True
