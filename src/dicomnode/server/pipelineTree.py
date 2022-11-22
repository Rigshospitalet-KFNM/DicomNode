"""_summary_
"""

__author__ = "Christoffer Vilstrup Jensen"

import logging
import shutil

from pydicom import Dataset
from pathlib import Path
from typing import Dict, Union, List, Optional, Type, Callable, Any


from dicomnode.lib.exceptions import InvalidRootDataDirectory, InvalidDataset
from dicomnode.lib.imageTree import ImageTreeInterface, IdentityMapping
from dicomnode.server.input import AbstractInput


class InputContainer(ImageTreeInterface):
  def __getitem__(self, key):
    if hasattr(self, 'instance'):
      return self.instance[key]
    else:
      raise KeyError("Instance not defined")

  def __init__(self,
               args: Dict[str, Type],
               container_path : Optional[str | Path],
               header_tags: List[int]= []
    ) -> None:
    if container_path is not None:
      container_path.mkdir(exist_ok=True)
    self.data: Dict[str, AbstractInput] = {}
    for arg_name, input in args.items():
      if container_path is not None:
        input_path = container_path / arg_name
      else:
        input_path = None
      self.data[arg_name] = input(input_path)

    self.images: int = 0

    # InputContainer tags:
    self.container_path = container_path
    self.header_tags: List[int] = header_tags

    #
    self.logger: logging.Logger = logging.getLogger("dicomnode")

  def _cleanup(self):
    if self.container_path:
      for input in self.data.values():
        input._clean_up()
      shutil.rmtree(self.container_path)

  def _validateAll(self):
    valid = True
    for input in self.data.values():
      valid &= input.validate()
    self.logger.debug(f"Validation returns: {valid}")
    return valid

  def get_header(self) -> Optional[Dataset]:
    if not hasattr(self, 'header'):
      return None
    return self.header

  def _get_data(self) -> 'InputContainer':
    new_instance = {}
    for arg_name, input in self.data.items():
      new_instance[arg_name] : input.get_data()
    self.instance = new_instance

    return self

  def add_image(self, dicom: Dataset) -> None:
    if not hasattr(self, 'header'):
      self.logger.debug("Adding Header")
      self.header = Dataset()
      for header_tag in self.header_tags:
        if header_tag in dicom:
          self.header[header_tag] = dicom[header_tag]
        else:
          self.logger("Could not add the add the header, rejecting image.")
          del self.header
          raise InvalidDataset()

    added = False
    for input in self.data.values():
      try:
        input.add_image(dicom)
        self.images += 1
        added = True
      except InvalidDataset:
        pass
    if not added:
      raise InvalidDataset()

  def map(self, func: Callable[[Dataset], Any], UIDMapping: Optional[IdentityMapping] = None) -> Dict[str, Any]:
    return super().map(func, UIDMapping)

class PipelineTree(ImageTreeInterface):
  """A more specialized ImageTree, which is used by a dicom node to keep track
  of studies.
  """

  def __init__(self,
               patient_identifier : int,
               pipelineArgs: Dict[str, Type], # Type of AbstractInputDataClass
               header_tags: List[int] =[],
               header_values: Dict[str, Any] = {},
               root_data_directory : Optional[Path] = None,
    ) -> None:
    """_summary_

    Args:
        root_data_directory (Path): _description_
        patient_identifier (int): _description_
        args (Dict[str, AbstractInputDataClass]): _description_
        dcm (Optional[Union[List[Dataset], Dataset]], optional): _description_. Defaults to None.
    """
    # ImageTreeInterface required attributes
    self.data: Dict[str, InputContainer] = {}
    self.images: int = 0

    # Args setup
    self.patient_identifier_tag: int = patient_identifier
    self.PipelineArgs: Dict[str, Type] = pipelineArgs
    self.root_data_directory: Optional[str | Path] = root_data_directory
    self.header_tags = header_tags
    self.header_values = header_values

    #Logger Setup
    self.logger: logging.Logger = logging.getLogger("dicomnode")

    #Load File state
    if self.root_data_directory is None: # There are no files to load if it's in memory
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
      self.logger.debug(f"{hex(self.patient_identifier_tag)} not in {dicom}") 
      self.logger.debug("Patient Identifier tag not in dicom")
      raise InvalidDataset()

    key = str(dicom[self.patient_identifier_tag].value)

    if key not in self.data:
      self.logger.debug("Patient not found in self data. Creating ")
      if self.root_data_directory is not None:
        patient_directory: Path = self.root_data_directory / key
        if patient_directory.exists():
          # TODO: It's possible to recover from this position, like done by some multithreading
          self.logger.error("Patient file directory exists while not part of pipelineTree")
          raise InvalidRootDataDirectory()
        else:
          patient_directory.mkdir()
      else:
        patient_directory = None

      patient_data = InputContainer(self.PipelineArgs, patient_directory, header_tags=self.header_tags)
      self.data[key] = patient_data
    else:
      patient_data = self.data[key]

    added = False
    for arg_name, IDC in patient_data.data.items():
      try:
        IDC.add_image(dicom)
        added = True
      except InvalidDataset:
        pass # The dataset doesn't fit here, so try another dataset

    if not added:
      raise InvalidDataset()
    else:
      self.images += 1

  def map(self,
          func: Callable[[Dataset], Any],
          UIDMapping: Optional[IdentityMapping] = None
    ) -> Dict[str, Any]:
    return super().map(func, UIDMapping)

  def validate_patient_ID(self, pid: str) -> Optional[InputContainer]:
    input_container = self.data[pid]
    if input_container._validateAll():
      return input_container._get_data()
    return None


  def remove_patient(self,patient_id: str):
    if patient_id in self.data:
      self.data[patient_id]._cleanup()
      del self.data[patient_id]
