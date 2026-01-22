"""This module defines all the different types of output for a pipeline

These pipeline all inherit from the pipeline output, which is just an interface
for exporting the images

"""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Any, List, Iterable, Tuple, Callable

# Third Party Packages
from pydicom import Dataset

# Dicomnode Packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.lib.io import save_dicom


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
               ae_title: str,
              ) -> None:
    self.ae_title = ae_title
    super().__init__(output)

  def send(self) -> bool:
    success = True
    logger = getLogger(DICOMNODE_LOGGER_NAME)

    for address, datasets in self:
      datasets_list = [dataset for dataset in datasets]
      try:
        logger.info(f"Sending {len(datasets_list)} datasets to {address.ae_title}")
        send_images(self.ae_title, address, datasets)
      except CouldNotCompleteDIMSEMessage:
        logger.error(f"Could not send to images to {address.ae_title}")
        success = False
    return success

class NoOutput(PipelineOutput):
  """This class represents, that the pipeline doesn't have any outputs
  useful when you test or you export your data in the process function call
  """
  output = []
  def __init__(self) -> None:
    super().__init__([])

  def send(self) -> bool:
    return True

def save_dataset(path, dataset) -> None:
  """Saves a data set at

  path / dataset.StudyInstanceUID.name
       / dataset.SeriesInstanceUID.name
       / dataset.SOPInstanceUID.name + .dcm

  Args:
      path (_type_): _description_
      dataset (_type_): _description_

  """
  dataset_path: Path = path / dataset.StudyInstanceUID.name \
                        / dataset.SeriesInstanceUID.name \
                        / (dataset.SOPInstanceUID.name + '.dcm')
  save_dicom(dataset_path,  dataset)


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

  output: Iterable[PipelineOutput]

  def __init__(self, outputs: Iterable[PipelineOutput]) -> None:
    self.outputs = outputs

  def send(self) -> bool:
    success = True

    for output in self.outputs:
      success &= output.send()

    return success


__all__ = [
  'PipelineOutput',
  'NoOutput',
  'DicomOutput',
  'FileOutput',
  'MultiOutput'
]