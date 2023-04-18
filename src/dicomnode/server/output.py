""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standart Library
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple, Type, Callable

# Third Party Packages
from pydicom import Dataset

# Dicomnode Packages
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.lib.dimse import Address, send_images
from dicomnode.lib.image_tree import DicomTree, ImageTreeInterface
from dicomnode.lib.io import save_dicom
from dicomnode.lib.logging import get_logger

logger = get_logger()

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
    """Method that should export all the data stored in the
    PipelineOutput

    Returns:
      bool: If the send was successful or not
    """
    raise NotImplementedError # pragma: no cover

  def __iter__(self):
    for destination, payload in self.output:
      yield destination, payload

class DicomOutput(PipelineOutput):
  """PipelineOutput that export dicom series via the DIMSE message
  protocol

  Args:
    output (List[Tuple[Address, Iterable[Dataset]]]) - A list of output
    ae_title (str): - SCU ae title

  """
  output: List[Tuple[Address, Iterable[Dataset]]]
  "Outputs to be send"

  def __init__(self,
               output: List[Tuple[Address, Iterable[Dataset]]],
               ae_title: str) -> None:
    self.ae = ae_title
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
  """PipelineOutput that saves series at a destination using the injected save_function

    Args:
      output (List[Tuple[Path, Iterable[Dataset]]]): A list of outputs
      saving_function (Callable[[Path, Dataset], None], optional):
        function that does the actual saving
        Defaults to save_dataset, which saves the datasets at:
        path / StudyInstanceUID / SeriesInstanceUID / SOP instanceUID

    Example:
    Saving the dicom series 'datasets_1' at path_1 and 'datasets_2' at path_2
    >>> file_output = FileOutput([(path_1, datasets_1),(path_2, datasets_2),])
    >>> file_output.send()
    True

  """

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

class MultiOutput(PipelineOutput):
  """This pipeline allows for multiple modality outputs

  Args:
    outputs (IterablePipelineOutput]): The outputs to send

  Example:
  Send datasets_1 to address_1 and address_2,
  datasets_2 to address_1, and store dataset_1 at path_1 and
  dataset_2 at path 2

  >>>multi_output = MultiOutput([DicomOutput([\
        (address_1, datasets_1),\
        (address_2, datasets_1),\
        (address_1, datasets_2),\
      ]),\
      FileOutput([\
        (path_1, datasets_1),(path_2, datasets_2),\
      ])\
    ])
  >>>multi_output.send()
  True
  """

  def __init__(self, outputs: Iterable[PipelineOutput]) -> None:
    self.outputs = outputs

  def send(self) -> bool:
    success = True

    for output in self.outputs:
      success &= output.send()

    return success
