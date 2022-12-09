"""_summary_
"""

__author__ = "Christoffer Vilstrup Jensen"

import logging
import shutil

from pydicom import Dataset
from pathlib import Path
from typing import Dict, Union, List, Optional, Type, Callable, Any


from dicomnode.lib.exceptions import InvalidRootDataDirectory, InvalidDataset, InvalidTreeNode
from dicomnode.lib.imageTree import ImageTreeInterface, IdentityMapping
from dicomnode.server.input import AbstractInput


class InputContainer(ImageTreeInterface):
  """This is the container containing all the series, of a study.

  """

  def __getitem__(self, key: str):
    if hasattr(self, 'instance'):
      return self.instance[key]
    else:
      return self.data[key]

  def __init__(self,
               args: Dict[str, Type],
               container_path : Optional[Path]
    ) -> None:
    super().__init__()

    if container_path is not None:
      if container_path.is_file():
        raise InvalidRootDataDirectory
      container_path.mkdir(exist_ok=True)

    for arg_name, input in args.items():
      if container_path is not None:
        input_path = container_path / arg_name
      else:
        input_path = None
      self.data[arg_name] = input(input_path)

    # InputContainer tags:
    self.container_path: Optional[Path] = container_path

    # logger
    self.logger: logging.Logger = logging.getLogger("dicomnode")

  def _cleanup(self):
    if self.container_path:
      for input in self.data.values():
        if isinstance(input, AbstractInput):
          input._clean_up()
        else:
          raise InvalidTreeNode # pragma: no cover
      shutil.rmtree(self.container_path)

  def _validateAll(self):
    valid = True
    for input in self.data.values():
      if isinstance(input, AbstractInput):
        valid &= input.validate()
      else:
        raise InvalidTreeNode # pragma: no cover
    self.logger.debug(f"Validation returns: {valid}")
    return valid

  def _get_data(self) -> 'InputContainer':
    new_instance: Dict[str, Any] = {}
    for arg_name, input in self.data.items():
      if isinstance(input, AbstractInput):
        new_instance[arg_name] = input.get_data()
      else:
        raise InvalidTreeNode # pragma: no cover
    self.instance = new_instance

    return self

  def add_image(self, dicom: Dataset) -> int:
    if not hasattr(self, 'header'):
      self.logger.debug("Adding Header")
      self.header = Dataset()

    added = 0
    for input in self.data.values():
      try:
        input.add_image(dicom)
        added += 1
      except InvalidDataset:
        pass
    if added == 0:
      raise InvalidDataset()
    self.images += added
    return added


class PipelineTree(ImageTreeInterface):
  """A more specialized ImageTree, which is used by a dicom node to keep track
  of studies.
  """
  def __init__(self,
               patient_identifier: int,
               pipelineArgs: Dict[str, Type], # Type of AbstractInputDataClass
               root_data_directory : Optional[Path] = None,
    ) -> None:
    """_summary_

    Args:
        patient_identifier (int): _description_
        pipelineArgs (Dict[str, Type]): _description_
        root_data_directory (Optional[Path], optional): _description_. Defaults to None.

    Raises:
        InvalidRootDataDirectory: _description_
    """
    # ImageTreeInterface required attributes
    super().__init__()

    # Args setup
    self.patient_identifier_tag: int = patient_identifier
    self.PipelineArgs: Dict[str, Type] = pipelineArgs
    self.root_data_directory: Optional[Path] = root_data_directory

    #Logger Setup
    self.logger: logging.Logger = logging.getLogger("dicomnode")

    #Load File state
    if self.root_data_directory is None: # There are no files to load if it's in memory
      return

    if not self.root_data_directory.exists():
      self.root_data_directory.mkdir()

    for patient_directory in self.root_data_directory.iterdir():
      if patient_directory.is_file():
        self.logger.error(f"{patient_directory.name} in root_data_directory is a file not a directory")
        raise InvalidRootDataDirectory()

      self[patient_directory.name] = InputContainer(self.PipelineArgs, patient_directory)

  def add_image(self, dicom : Dataset) -> int:
    if self.patient_identifier_tag not in dicom:
      self.logger.debug(f"{hex(self.patient_identifier_tag)} not in dataset")
      self.logger.debug("Patient Identifier tag not in dicom")
      raise InvalidDataset()

    key = str(dicom[self.patient_identifier_tag].value)

    if key not in self:
      IDC_path: Optional[Path] = None
      if self.root_data_directory is not None:
        IDC_path = self.root_data_directory / key
      self[key] = InputContainer(self.PipelineArgs, IDC_path)

    IDC = self[key]
    if isinstance(IDC, InputContainer):
      added = IDC.add_image(dicom)
      self.images += added
      return added
    else:
      raise InvalidTreeNode # pragma: no cover


  def validate_patient_ID(self, pid: str) -> Optional[InputContainer]:
    input_container = self[pid]
    if input_container is None:
      return None
    elif isinstance(input_container, InputContainer):
      if input_container._validateAll():
        return input_container._get_data()
      return None
    else:
      raise InvalidTreeNode # pragma: no cover

  def remove_patient(self,patient_id: str) -> None:
    if patient_id in self:
      IC = self[patient_id]
      if isinstance(IC, InputContainer):
        IC._cleanup()
        del self[patient_id]
      else:
        raise InvalidTreeNode # pragma: no cover
