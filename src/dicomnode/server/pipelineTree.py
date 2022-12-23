"""_summary_
"""

__author__ = "Christoffer Vilstrup Jensen"

import logging
from logging import Logger
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydicom import Dataset

from dicomnode.lib.dicomFactory import DicomFactory, SeriesHeader, Blueprint
from dicomnode.lib.exceptions import (InvalidDataset, InvalidRootDataDirectory,
                                      InvalidTreeNode)
from dicomnode.lib.imageTree import ImageTreeInterface
from dicomnode.server.input import AbstractInput

class InputContainer:
  def __init__(self, data: Dict[str, Any], header: Optional[SeriesHeader] = None, paths: Optional[Dict[str, Path]] = None) -> None:
    self.__data = data
    self.header = header
    self.paths  = paths

  def __getitem__(self, key: str):
    return self.__data[key]


class PatientNode(ImageTreeInterface):
  """This is the container containing all the Inputs of your pipeline."""

  @dataclass
  class Options:
    logger: Optional[Logger] = None
    container_path: Optional[Path] = None
    factory: Optional[DicomFactory] = None
    header_blueprint: Optional[Blueprint] = None
    lazy: bool = False


  def __init__(self,
               args: Dict[str, Type[AbstractInput]],
               pivot: Optional[Dataset] = None, # See lazyness
               options: Options = Options(),
    ) -> None:
    super().__init__()
    self.options = options

    if self.options.container_path is not None:
      if self.options.container_path.is_file():
        raise InvalidRootDataDirectory
      self.options.container_path.mkdir(exist_ok=True)

    if self.options.factory is not None and pivot is not None and self.options.header_blueprint is not None:
      self.header = self.options.factory.make_series_header(pivot, self.options.header_blueprint)
    else:
      self.header = None


    for arg_name, input in args.items():
      input_path: Optional[Path] = None
      if self.options.container_path is not None:
        input_path = self.options.container_path / arg_name

      inputOptions = self.__get_Input_Options(input=input, input_path=input_path)

      self.data[arg_name] = input(pivot, options=inputOptions)

    # logger
    if self.options.logger is not None:
      self.logger = self.options.logger
    else:
      self.logger = logging.getLogger("dicomnode")

  def _cleanup(self):
    if self.options.container_path is not None:
      for input in self.data.values():
        if isinstance(input, AbstractInput):
          input._clean_up()
        else:
          raise InvalidTreeNode # pragma: no cover
      shutil.rmtree(self.options.container_path)

  def _validateAll(self):
    valid = True
    for input in self.data.values():
      if isinstance(input, AbstractInput):
        valid &= input.validate()
      else:
        raise InvalidTreeNode # pragma: no cover
    return valid

  def _get_data(self) -> InputContainer:
    new_instance: Dict[str, Any] = {}
    paths: Optional[Dict[str, Path]]
    if self.options.container_path is None:
      paths = None
    else:
      paths = {}
    for arg_name, input in self.data.items():
      if isinstance(input, AbstractInput):
        new_instance[arg_name] = input.get_data()
        if paths is not None and input.path is not None:
          paths[arg_name] = input.path
      else:
        raise InvalidTreeNode # pragma: no cover
    input_container = InputContainer(new_instance, self.header, paths)

    return input_container

  def add_image(self, dicom: Dataset) -> int:
    if not hasattr(self, 'header') and self.options.factory is not None and self.options.header_blueprint is not None:
      self.logger.debug("Adding Header")
      self.header = self.options.factory.make_series_header(dicom, self.options.header_blueprint)

    added = 0
    for input in self.data.values():
      if isinstance(input, AbstractInput):
        try:
          added += input.add_image(dicom)
        except InvalidDataset:
          pass
      else:
        raise InvalidTreeNode # pragma: no cover
    if added == 0:
      raise InvalidDataset()
    self.images += added
    return added

  def __get_Input_Options(self, input: Type[AbstractInput], input_path: Optional[Path]):
    return input.Options(
        data_directory = input_path,
        logger=self.options.logger,
        factory = self.options.factory,
        lazy=self.options.lazy
      )

class PipelineTree(ImageTreeInterface):
  """A more specialized ImageTree, which is used by a dicom node to keep track
  of studies.
  """

  @dataclass
  class Options:
    input_container: type[PatientNode] = PatientNode
    data_directory: Optional[Path] = None
    factory: Optional[DicomFactory] = None
    lazy: bool = False

  def __init__(self,
               patient_identifier: int,
               pipelineArgs: Dict[str, Type[AbstractInput]], # Type of AbstractInputDataClass
               options = Options()
    ) -> None:
    """_summary_

    Args:
        patient_identifier (int): _description_
        pipelineArgs (Dict[str, Type]): _description_
        Options: PipelineTree.Options

    Raises:
        InvalidRootDataDirectory: _description_
    """
    # ImageTreeInterface required attributes
    super().__init__()

    # Args setup
    self.patient_identifier_tag: int = patient_identifier
    self.PipelineArgs: Dict[str, Type[AbstractInput]] = pipelineArgs
    self.root_data_directory: Optional[Path] = options.data_directory
    self.options = options

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

      options = self.__get_InputContainer_Options(patient_directory)

      self[patient_directory.name] = PatientNode(self.PipelineArgs, None, options)

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

      options = self.__get_InputContainer_Options(IDC_path)
      self[key] = PatientNode(self.PipelineArgs, dicom, options)

    IDC = self[key]
    if isinstance(IDC, PatientNode):
      added = IDC.add_image(dicom)
      self.images += added
      return added
    else:
      raise InvalidTreeNode # pragma: no cover


  def validate_patient_ID(self, pid: str) -> Optional[InputContainer]:
    input_container = self[pid]
    if input_container is None:
      return None
    elif isinstance(input_container, PatientNode):
      if input_container._validateAll():
        self.logger.debug(f"sufficient data for patient {pid}")
        return input_container._get_data()
      self.logger.debug(f"insufficient data for patient {pid}")
      return None
    else:
      raise InvalidTreeNode # pragma: no cover

  def remove_patient(self,patient_id: str) -> None:
    if patient_id in self:
      IC = self[patient_id]
      if isinstance(IC, PatientNode):
        IC._cleanup()
        del self[patient_id]
      else:
        raise InvalidTreeNode # pragma: no cover

  def __str__(self) -> str:
    return str(self.data)

  def __get_InputContainer_Options(self, container_path: Optional[Path]) -> PatientNode.Options:
    """Creates the options for the underlying Input Container

    Args:
        container_path (Optional[Path]): _description_

    Returns:
        InputContainer.Options: _description_
    """

    return PatientNode.Options(
        container_path=container_path,
        factory=self.options.factory,
        logger=self.logger,
        lazy=self.options.lazy
      )
