"""This module is the data structure of a servers dicom storage.



"""

__author__ = "Christoffer Vilstrup Jensen"

import logging
from logging import Logger
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union, Iterable

from datetime import datetime

from pydicom import Dataset

from dicomnode.lib.dicomFactory import DicomFactory, SeriesHeader, Blueprint, FillingStrategy
from dicomnode.lib.exceptions import (InvalidDataset, InvalidRootDataDirectory,
                                      InvalidTreeNode)
from dicomnode.lib.imageTree import ImageTreeInterface
from dicomnode.server.input import AbstractInput, DynamicInput, DynamicLeaf

def __type_check_list(dicom_list: List[Dataset], maybe_dicom_iterator: Iterable[Any]):
  for maybe_dicom in maybe_dicom_iterator:
    if isinstance(maybe_dicom, Dataset):
      dicom_list.append(maybe_dicom)
    else:
      raise InvalidTreeNode

def __extract_pivot_list(abstract_input: AbstractInput) -> List[Dataset]:
  return_list: List[Dataset] = []
  if isinstance(abstract_input, DynamicInput):
    for dynamic_leaf in abstract_input.data.values():
      if isinstance(dynamic_leaf, DynamicLeaf):
        __type_check_list(return_list, dynamic_leaf)
      else:
        raise InvalidTreeNode # pragma: no cover
      break
  else:
    __type_check_list(return_list, abstract_input)

  return return_list


class InputContainer:
  """Simple container class for grinded input.
  """
  def __init__(self, data: Dict[str, Any], header: Optional[SeriesHeader] = None, paths: Optional[Dict[str, Path]] = None) -> None:
    self.__data = data
    self.header = header
    self.paths  = paths

  def __getitem__(self, key: str):
    return self.__data[key]


class PatientNode(ImageTreeInterface):
  """This is an ImageTree node, where each subnode contains related images.
  When filled can run through the pipelines process function.
  The Node contains AbstractInputs and only AbstractInput

  This class is responsible for:
    - Determine if there's sufficient data to run the pipeline processing
    - Initializing a SeriesHeader
    - Creating InputContainer
  """
  #In most cases, each subnode is also a leaf.

  @dataclass
  class Options:
    ae_title: Optional[str] = None
    container_path: Optional[Path] = None
    factory: Optional[DicomFactory] = None
    lazy: bool = False
    logger: Optional[Logger] = None
    header_blueprint: Optional[Blueprint] = None
    filling_strategy: FillingStrategy = FillingStrategy.DISCARD
    InputContainerType: Type[InputContainer] = InputContainer
    pivot_input: Optional[str] = None


  def __init__(self,
               args: Dict[str, Type[AbstractInput]],
               pivot: Optional[Dataset] = None, # See lazyness
               options: Options = Options(),
    ) -> None:
    super().__init__()
    self.options = options
    self.creationTime = datetime.now()

    if self.options.container_path is not None:
      if self.options.container_path.is_file():
        raise InvalidRootDataDirectory
      self.options.container_path.mkdir(exist_ok=True)

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
    """Retrieved data in the way it's supposed to be processed in.

    Raises:
        InvalidTreeNode: _description_

    Returns:
        InputContainer: _description_
    """
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
    if self.options.factory is not None and self.options.header_blueprint is not None:
      pivot_list: List[Dataset] = []

      if self.options.pivot_input is not None:
        pivot_input= self.data[self.options.pivot_input]
        if isinstance(pivot_input, AbstractInput):
          pivot_list = __extract_pivot_list(pivot_input)
        else:
          raise InvalidTreeNode # pragma: no cover
      else:
        for abstract_input in self.data.values():
          if isinstance(abstract_input, AbstractInput):
            pivot_list = __extract_pivot_list(abstract_input)
          else:
            raise InvalidTreeNode
          break

      header = self.options.factory.make_series_header(
        pivot_list, self.options.header_blueprint, self.options.filling_strategy
      )
    else:
      header = None

    input_container = self.options.InputContainerType(new_instance, header, paths)

    return input_container

  def add_image(self, dicom: Dataset) -> int:
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
        ae_title=self.options.ae_title,
        data_directory = input_path,
        logger=self.options.logger,
        factory = self.options.factory,
        lazy=self.options.lazy
      )

class PipelineTree(ImageTreeInterface):
  """A more specialized ImageTree, which is used by a dicom node to keep track
  of studies.

  This is the root node of the servers ImageTree, It contains leafs of PatientNodes and only PatientNodes.

  The main responsibility of this class is to manage Patient Nodes: So
    - Creating new PatientNodes when needed
    - Deleting PatientNodes, when they're expired or processed
  """

  @dataclass
  class Options:
    ae_title: Optional[str] = None
    data_directory: Optional[Path] = None
    factory: Optional[DicomFactory] = None
    lazy: bool = False
    filling_strategy: FillingStrategy = FillingStrategy.DISCARD
    input_container_type: Type[InputContainer] = InputContainer
    patient_container: Type[PatientNode] = PatientNode
    header_blueprint: Optional[Blueprint] = None

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

      options = self.__get_PatientContainer_Options(patient_directory)

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

      options = self.__get_PatientContainer_Options(IDC_path)
      self[key] = PatientNode(self.PipelineArgs, dicom, options)

    IDC = self[key]
    if isinstance(IDC, PatientNode):
      added = IDC.add_image(dicom)
      self.images += added
      return added
    else:
      raise InvalidTreeNode # pragma: no cover


  def validate_patient_ID(self, pid: str) -> Optional[InputContainer]:
    """Determines if a patient have all needed data and extract it if it does

    Args:
      pid (str): patient to be validated

    Raises:
      InvalidTreeNode: If value at patient id is not a PatientNode

    Returns:
      Optional[InputContainer]: If there's insufficient data returns None,
        Otherwise a InputContainer with the grinded values
    """
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

  def remove_expired_studies(self, expiry_time : datetime):
    """Removes any study in the tree that have expired.

    Args:
      expiry_time (datetime): Any study created before expiry_time is considered to be expired.
  
    Raises:
      InvalidTreeNode: If a node is not a PatientNode
    """
    for patient_id, patient_node in self.data:
      if not isinstance(patient_node, PatientNode):
        raise InvalidTreeNode # pragma: no cover
      if patient_node.creationTime < expiry_time:
        self.remove_patient(patient_id)


  def remove_patient(self, patient_id: str) -> None:
    """Removes a patient from the tree

    Args:
        patient_id (str): identifier of the patient to be deleted

    Raises:
        InvalidTreeNode: If value at patient id is not a PatientNode
    """
    if patient_id in self:
      patient_node = self[patient_id]
      if isinstance(patient_node, PatientNode):
        patient_node._cleanup()
        del self[patient_id]
      else:
        raise InvalidTreeNode # pragma: no cover


  def __get_PatientContainer_Options(self, container_path: Optional[Path]) -> PatientNode.Options:
    """Creates the options for the underlying Patient Container

    Args:
        container_path (Optional[Path]): patient container path. Needed as input since path is per container

    Returns:
        PatientContainer.Options: Options ready to be injected.
    """

    return self.options.patient_container.Options(
        ae_title=self.options.ae_title,
        container_path=container_path,
        factory=self.options.factory,
        logger=self.logger,
        lazy=self.options.lazy,
        InputContainerType=self.options.input_container_type,
        header_blueprint=self.options.header_blueprint,
        filling_strategy=self.options.filling_strategy
      )
