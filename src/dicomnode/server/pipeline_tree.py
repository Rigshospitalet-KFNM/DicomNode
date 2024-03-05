"""This module is the data structure of a servers dicom storage.



"""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Type, Iterable, Set

# Third Party Python Packages
from pydicom import Dataset

# Dicomnode Library Packages
from dicomnode.dicom.dicom_factory import DicomFactory, SeriesHeader, Blueprint, FillingStrategy
from dicomnode.dicom.dimse import Address
from dicomnode.lib.exceptions import (InvalidDataset, InvalidRootDataDirectory,
                                      InvalidTreeNode, HeaderConstructionFailure)
from dicomnode.data_structures.image_tree import ImageTreeInterface
from dicomnode.lib.logging import log_traceback, get_logger
from dicomnode.server.input import AbstractInput, DynamicInput, DynamicLeaf

class InputContainer:
  """Simple container class for grinded input.
  """
  responding_address: Optional[Address] = None

  def __init__(self,
               data: Dict[str, Any],
               header: Optional[SeriesHeader] = None,
               paths: Optional[Dict[str, Path]] = None) -> None:
    self.__data = data
    self.header = header
    self.paths  = paths

  def __getitem__(self, key: str):
    if self.__data is None:
      raise KeyError(key)
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
    InputContainerType: Type[InputContainer] = InputContainer
    "Type that this node should return from a extract_input_container call"


    container_path: Optional[Path] = None
    """Path to permanent storage
    If None no permanent storage will be
    """

    ae_title: Optional[str] = None
    ""
    lazy: bool = False
    ""
    logger: Optional[Logger] = None
    """Logger that this object uses for any logs
    If None it will create a 'dicomnode' logger and log to it
    Injected Into AbstractInputs
    """

    # Dicom Header creation
    factory: Optional[DicomFactory] = None
    """Factory that should produce the series header, if None produce
    no Series header
    Injected Into AbstractInputs
    """
    header_blueprint: Optional[Blueprint] = None
    """The blueprint that the series header should be created from,
    if None no SeriesHeader will produced.
    Injected Into AbstractInputs
    """
    filling_strategy: FillingStrategy = FillingStrategy.DISCARD
    """Strategy that DicomFactory will follow, when creating SeriesHeader
    Injected Into AbstractInputs
    """
    parent_input: Optional[str] = None
    """The input that will be used as parent input in header creation,
    if None an arbitrary input is used.
    Injected Into AbstractInputs"""
    """Options for a patient node
    """
    ae_title: Optional[str] = None
    container_path: Optional[Path] = None
    factory: Optional[DicomFactory] = None
    lazy: bool = False
    logger: Optional[Logger] = None
    header_blueprint: Optional[Blueprint] = None
    filling_strategy: FillingStrategy = FillingStrategy.DISCARD
    input_container_type: Type[InputContainer] = InputContainer
    pivot_input: Optional[str] = None


  def __init__(self,
               args: Dict[str, Type[AbstractInput]],
               pivot: Optional[Dataset] = None, # See lazyness
               options: Options = Options(),
    ) -> None:
    super().__init__()
    self.options = options
    self.creation_time = datetime.now()

    if self.options.container_path is not None:
      if self.options.container_path.is_file():
        raise InvalidRootDataDirectory
      self.options.container_path.mkdir(exist_ok=True)

    for arg_name, dicomnode_input_type in args.items():
      input_path: Optional[Path] = None
      if self.options.container_path is not None:
        input_path = self.options.container_path / arg_name

      input_options = self._get_input_options(dicomnode_input=dicomnode_input_type,
                                              input_path=input_path)

      self.data[arg_name] = dicomnode_input_type(options=input_options)

    # logger
    if self.options.logger is not None:
      self.logger = self.options.logger
    else:
      self.logger = get_logger()

  # Helper function that must be class functions as name mangling fucks them up
  # if they were just private module functions
  def _type_check_list(self,
                        dicom_list: List[Dataset],
                        maybe_dicom_iterator: Iterable[Any]) -> None:
    for maybe_dicom in maybe_dicom_iterator:
      if isinstance(maybe_dicom, Dataset):
        dicom_list.append(maybe_dicom)
      else:
        raise InvalidTreeNode #pragma: no cover

  def __extract_pivot_list(self,abstract_input: AbstractInput) -> List[Dataset]:
    return_list: List[Dataset] = []
    if isinstance(abstract_input, DynamicInput):
      for dynamic_leaf in abstract_input.data.values():
        if isinstance(dynamic_leaf, DynamicLeaf):
          self._type_check_list(return_list, dynamic_leaf)
        else:
          raise InvalidTreeNode # pragma: no cover
        break
    else:
      self._type_check_list(return_list, abstract_input)

    return return_list


  def clean_up(self) -> int:
    """This function cleans up the patient node and owned all inputs

    Raises:
        InvalidTreeNode: _description_
    """
    images_removed = 0
    for dicomnode_input in self.data.values():
      if isinstance(dicomnode_input, AbstractInput):
        images_removed += dicomnode_input.clean_up()
      else:
        raise InvalidTreeNode # pragma: no cover
    if self.options.container_path is not None:
      shutil.rmtree(self.options.container_path)
    return images_removed

  def validate_inputs(self) -> bool:
    valid = True
    for dicomnode_input in self.data.values():
      if isinstance(dicomnode_input, AbstractInput):
        valid &= dicomnode_input.validate()
      else:
        raise InvalidTreeNode # pragma: no cover
    return valid

  def extract_input_container(self) -> InputContainer:
    """Retrieved inputs' data in the way it's supposed to be processed in.

    Raises:
        InvalidTreeNode: _description_

    Returns:
        InputContainer: _description_
    """
    data_directory: Dict[str, Any] = {}

    path_directory: Optional[Dict[str, Path]]
    if self.options.container_path is None:
      path_directory = None
    else:
      path_directory = {}

    for arg_name, dicomnode_input in self.data.items():
      if isinstance(dicomnode_input, AbstractInput):
        self.logger.debug(f"Extracting input from {dicomnode_input.__class__.__name__}")
        data_directory[arg_name] = dicomnode_input.get_data()
        if path_directory is not None and dicomnode_input.path is not None:
          path_directory[arg_name] = dicomnode_input.path
      else:
        raise InvalidTreeNode # pragma: no cover

    self.logger.debug("Extracted data from all inputs")

    if self.options.factory is not None and self.options.header_blueprint is not None:
      pivot_list: List[Dataset] = []

      if self.options.parent_input is not None:
        pivot_input= self.data[self.options.parent_input]
        if isinstance(pivot_input, AbstractInput):
          pivot_list = self.__extract_pivot_list(pivot_input)
        else:
          raise InvalidTreeNode # pragma: no cover
      else:
        for pivot_input in self.data.values():
          if isinstance(pivot_input, AbstractInput):
            pivot_list = self.__extract_pivot_list(pivot_input)
          else:
            raise InvalidTreeNode # pragma: no cover
          break

      try:
        header = self.options.factory.make_series_header(
          pivot_list, self.options.header_blueprint, self.options.filling_strategy
        )
      except HeaderConstructionFailure as exception:
        log_traceback(self.logger, exception, header_message="Failed to construct header")
        raise exception
      self.logger.debug("Successfully Constructed Header")

    else:
      self.logger.debug(f"Not Constructing header\nFactory is None: \
                        {self.options.factory is None}\nBlueprint is None: \
                        {self.options.header_blueprint is None}")

      header = None

    input_container = self.options.input_container_type(data_directory, header, path_directory)

    return input_container

  def add_image(self, dicom: Dataset) -> int:
    added = 0
    for dicomnode_input in self.data.values():
      if isinstance(dicomnode_input, AbstractInput):
        try:
          added += dicomnode_input.add_image(dicom)
        except InvalidDataset:
          # The dataset doesn't belong here
          # You know I could just check if return value is 0 to get the same
          # information. Since this just makes the Exceptional normal, i.e.
          # code smell
          pass
      else:
        raise InvalidTreeNode # pragma: no cover
    if added == 0:
      raise InvalidDataset()
    self.images += added
    return added

  def _get_input_options(self, dicomnode_input: Type[AbstractInput], input_path: Optional[Path]):
    return dicomnode_input.Options(
        ae_title=self.options.ae_title,
        data_directory = input_path,
        logger=self.options.logger,
        factory = self.options.factory,
        lazy=self.options.lazy
      )


class PipelineTree(ImageTreeInterface):
  """A more specialized ImageTree, which is used by a dicom node to keep track
  of studies.

  This is the root node of the servers ImageTree,
  It contains leafs of PatientNodes and only PatientNodes.
  Most function will raise the InvalidTreeNode exception if this is violated.

  The main responsibility of this class is to manage Patient Nodes: So
    - Creating new PatientNodes when needed
    - Deleting PatientNodes, when they're expired or processed
  """

  @dataclass
  class Options:
    """Options for a Pipeline Tree"""

    # This class mainly exists to have it's options injected to it from on high
    # rather that having all of these thing parse into it from args
    ae_title: Optional[str] = None
    "AE title of node"

    data_directory: Optional[Path] = None
    "Root directory for file storage"

    factory: Optional[DicomFactory] = None
    "DicomFactory for constructing Series Header"

    filling_strategy: FillingStrategy = FillingStrategy.DISCARD
    "Filling Strategy for 'factory'"

    header_blueprint: Optional[Blueprint] = None
    "Blueprint for factory to use"

    lazy: bool = False
    "If underlying inputs should use lazy datasets"

    logger: Optional[Logger] = None
    "Logger to send message to"

    input_container_type: Type[InputContainer] = InputContainer
    "type of input containers get_patient_input_container should return"

    parent_input: Optional[str] = None
    "Input, that should be used as parent"

    patient_container: Type[PatientNode] = PatientNode
    "Type of node that's under this tree."


  def __init__(self,
               patient_identifier: int,
               pipelineArgs: Dict[str, Type[AbstractInput]],
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
    self.patient_identifier_tag: int = patient_identifier # Move this to options?
    self.tree_node_definition: Dict[str, Type[AbstractInput]] = pipelineArgs
    self.options = options

    self._locked_patients: Set[str] = set()

    #Logger Setup
    if self.options.logger is None:
      self.logger = get_logger()
    else:
      self.logger = self.options.logger

    #Load File state
    if self.options.data_directory is None:
      # There are no files to load if it's in memory
      return

    if not self.options.data_directory.exists():
      self.options.data_directory.mkdir()

    for patient_directory in self.options.data_directory.iterdir():
      if patient_directory.is_file():
        self.logger.error(f"{patient_directory.name} in\
                           root_data_directory is a file not a directory")
        raise InvalidRootDataDirectory()

      options = self._get_patient_container_options(patient_directory)

      self[patient_directory.name] = PatientNode(self.tree_node_definition, None, options)

  def add_image(self, dicom : Dataset) -> int:
    key = self.get_patient_id(dicom)

    if key in self._locked_patients:
      self.logger.error("Attempting to add an image to a locked patient")
      raise InvalidDataset()

    if key not in self:
      input_container_path: Optional[Path] = None
      if self.options.data_directory is not None:
        input_container_path = self.options.data_directory / key

      options = self._get_patient_container_options(input_container_path)
      self[key] = PatientNode(self.tree_node_definition, dicom, options)

    patient_node = self[key]
    if isinstance(patient_node, PatientNode):
      added = patient_node.add_image(dicom)
      self.images += added
      return added

    raise InvalidTreeNode # pragma: no cover


  def validate_patient_id(self, patient_id: str) -> bool:
    """Determines if a patient have all needed data and extract it if it does

    Args:
      patient_id (str): Patient reference

    Raises:
      InvalidTreeNode: If value at patient id is not a PatientNode

    Returns:
      Optional[InputContainer]: If there's insufficient data returns None,
        Otherwise a InputContainer with the grinded values
    """
    patient_node = self[patient_id]
    if isinstance(patient_node, PatientNode):
      return patient_node.validate_inputs()

    raise InvalidTreeNode # pragma: no cover

  def get_patient_id(self, dataset: Dataset) -> str:
    """Retrieves the values that will be used as key in the Tree for the node
    That represents the input dataset

    Args:
      dataset (Dataset): the dataset, where the key will be extracted

    Raises:
      InvalidDataset: If the value used for separating is missing, or is None
    """
    if self.patient_identifier_tag not in dataset:
      self.logger.debug(f"{hex(self.patient_identifier_tag)} not in dataset")
      self.logger.debug("Patient Identifier tag not in dataset")
      raise InvalidDataset()

    value = dataset[self.patient_identifier_tag].value

    if value is None:
      self.logger.debug(f"Input dataset have tag {hex(self.patient_identifier_tag)}\
                         but it's None and therefore unhashable")
      raise InvalidDataset()

    return str(value)

  def get_patient_input_container(self, patient_id: str) -> InputContainer:
    """Gets the input container with the associated patient.
    Assumes that patient id is in self

    Args:
        patient_id (str): The Patient that we want data from

    Returns:
        InputContainer: InputContainer with preprocessed data

    Raises:
      KeyError: If patient id is not in self
      InvalidTreeNode: If value at patient_id is not a PatientNode instance
    """
    patient_node = self[patient_id]

    self.logger.debug(f"Getting Patient node: {patient_id}")
    if isinstance(patient_node, PatientNode):
      return patient_node.extract_input_container()

    # Pipeline Tree Patient node constraint violated!
    raise InvalidTreeNode # pragma: no cover


  def remove_expired_studies(self, expiry_time : datetime):
    """Removes any PatientNode in the tree that have expired.

    Args:
      expiry_time (datetime): Any study created before expiry_time is considered to be expired.
    Raises:
      InvalidTreeNode: If a node is not a PatientNode
    """
    # Note here that iteration over a dict, require the dict not to change
    # Hence the need to two loops.
    # Note that this is a multi threaded application and there is a race condition
    # if this function runs while another thread is iteration over the data dir.
    to_be_removed = set()
    for patient_id, patient_node in self.data.items():
      if isinstance(patient_node, PatientNode):
        if patient_node.creation_time < expiry_time:
          to_be_removed.add(patient_id)
      else:
        raise InvalidTreeNode #pragma: no cover

    self.clean_up_patients(to_be_removed)

  def clean_up_patients(self, patient_ids: Iterable[str]):
    """Removes many patients from the pipeline tree

    Args:
        patient_ids (Iterable[str]): Collection of patient ids to be removed.

    Raises:
        InvalidTreeNode: If nodes are not PatientNodes
    """
    #Due to the fact, that you cannot iterator over a changing directory

    new_data_dict = {}
    removed_images = 0

    for patient_id, patient_node in self.data.items():
      if patient_id in patient_ids:
        if isinstance(patient_node, PatientNode):
          removed_images += patient_node.clean_up()
        else:
          raise InvalidTreeNode # pragma: no cover
      else:
        new_data_dict[patient_id] = patient_node

    for patient_id in patient_ids:
      self._locked_patients.discard(patient_id)
    self.images -= removed_images
    self.data = new_data_dict
    self.logger.debug(f"Removed {removed_images} from {len(patient_ids)} Patients")

  def clean_up_patient(self, patient_id: str) -> None:
    """Removes a patient from the tree, and removes any files stored under the patient

    Args:
        patient_id (str): identifier of the patient to be deleted

    Raises:
        InvalidTreeNode: If value at patient id is not a PatientNode
    """

    # This function is a tad weird, since it needs to be thread safe.
    new_data_dict = {}
    removed_images = 0

    for patient_id_dict, patient_node in self.data.items():
      if patient_id_dict == patient_id:
        if isinstance(patient_node, PatientNode):
          removed_images = patient_node.clean_up()
        else:
          raise InvalidTreeNode # pragma: no cover
      else:
        new_data_dict[patient_id_dict] = patient_node


    self.images -= removed_images
    self.data = new_data_dict
    self._locked_patients.discard(patient_id)
    self.logger.debug(f"Removed {patient_id} and {removed_images} images from Pipeline")

  def lock_patient(self, patient_id):
    self._locked_patients.add(patient_id)

  def _get_patient_container_options(self, container_path: Optional[Path]) -> PatientNode.Options:
    """Creates the options for the underlying Patient Container

    Args:
        container_path (Optional[Path]): patient container path. Needed as
                                         input since path is per container

    Returns:
        PatientContainer.Options: Options ready to be injected.
    """

    return self.options.patient_container.Options(
        ae_title=self.options.ae_title,
        container_path=container_path,
        factory=self.options.factory,
        logger=self.logger,
        lazy=self.options.lazy,
        input_container_type=self.options.input_container_type,
        header_blueprint=self.options.header_blueprint,
        filling_strategy=self.options.filling_strategy
      )
