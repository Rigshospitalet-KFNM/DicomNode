"""_summary_
"""

__author__ = "Christoffer Vilstrup Jensen"

import logging
from pydicom import Dataset
from pathlib import Path
from typing import Dict, Union, List, Optional, Type, Callable, Any

from dicomnode.lib.exceptions import InvalidRootDataDirectory, InvalidDataset
from dicomnode.server.input import AbstractInput
from dicomnode.lib.imageTree import ImageTreeInterface


class PipelineTree(ImageTreeInterface):
  """A more specialized ImageTree, which is used by a dicom node to keep track
  of studies.
  """

  def __init__(self,
               root_data_directory : Path,
               patient_identifier : int,
               pipelineArgs: Dict[str, Type], # Type of AbstractInputDataClass
               in_memory: bool = False,
    ) -> None:
    """_summary_

    Args:
        root_data_directory (Path): _description_
        patient_identifier (int): _description_
        args (Dict[str, AbstractInputDataClass]): _description_
        dcm (Optional[Union[List[Dataset], Dataset]], optional): _description_. Defaults to None.
    """
    self.logger: logging.Logger = logging.getLogger("dicomnode")
    self.data: Dict[str, Dict[str, AbstractInput]] = {}
    self.images: int = 0
    self.patient_identifier_tag: int = patient_identifier
    self.PipelineArgs: Dict[str, Type] = pipelineArgs
    self.root_data_directory: str | Path = root_data_directory
    self.in_memory: bool = in_memory

    #Load File state
    if self.in_memory: # There are no files to load if it's in memory
      return

    for patient_directory in self.root_data_directory.iterdir():
      if patient_directory.is_file():
        self.logger.error(f"{patient_directory.name} in root_data_directory is a file not a directory")
        raise InvalidRootDataDirectory()

      patient_data: Dict[str, AbstractInput] = {}
      for arg_name, IDC in self.PipelineArgs.items():
        IDC_path: Path = patient_directory / arg_name
        if not IDC_path.exists():
          self.logger.error(f"Patient {patient_directory.name}'s {IDC_path.name} doesn't exists")
          raise InvalidRootDataDirectory()
        if IDC_path.is_file():
          self.logger.error(f"Patient {patient_directory.name}'s {IDC_path.name} is a file not a directory")
          raise InvalidRootDataDirectory()
        patient_data[arg_name] = IDC(IDC_path)

      self.data[patient_directory.name] = patient_data

  def add_image(self, dicom : Dataset) -> None:
    if self.patient_identifier_tag not in dicom:
      raise InvalidDataset()

    key = str(dicom[self.patient_identifier_tag].value)


    if key not in self.data:
      if not self.in_memory:
        patient_directory: Path = self.root_data_directory / key
        if patient_directory.exists():
          # TODO: It's possible to recover from this position, like done by some multithreading
          self.logger.error("Patient file directory exists while not part of pipelineTree")
          raise InvalidRootDataDirectory()
        else:
          patient_directory.mkdir()

      patient_data: Dict[str, AbstractInput] = {}
      for arg_name, IDC in self.PipelineArgs.items():
        IDC_path = None
        if not self.in_memory:
          IDC_path = patient_directory / arg_name
          IDC_path.mkdir()
        patient_data[arg_name] = IDC(IDC_path, in_memory=self.in_memory)
      self.data[key] = patient_data

    added = False
    for arg_name, IDC in self.data[key].items():
      try:
        IDC.add_image(dicom)
        added = True
      except InvalidDataset:
        pass # The dataset doesn't fit here, so try another dataset

    if not added:
      raise InvalidDataset()
    else:
      self.images += 1

  def map(self, func: Callable[[Dataset], Any], UIDMapping) -> Any:
    pass

  def validate_patient_ID(self, pid: str) -> Optional[Dict[str, Any]]:
    valid = True
    for input in self.data[pid].values():
      valid &= input.validate()

    if valid:
      return_data = {}
      for arg_name, input in self.data[pid].items():
        return_data[arg_name] = input.get_data()

