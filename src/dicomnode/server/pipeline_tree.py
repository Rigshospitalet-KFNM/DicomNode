"""This module is the data structure of a servers dicom storage.



"""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from dataclasses import dataclass
from datetime import datetime, date
from logging import Logger, getLogger
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Type, Iterable, Set

# Third Party Python Packages
from pydicom import Dataset

# Dicomnode Library Packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.dicom.series import DicomSeries
from dicomnode.data_structures.image_tree import ImageTreeInterface
from dicomnode.dicom.dimse import Address
from dicomnode.lib.io import Directory
from dicomnode.lib.exceptions import InvalidDataset, InvalidRootDataDirectory,\
                                      InvalidTreeNode

from dicomnode.server.input import AbstractInput

class InputContainer:
  """Simple container class for grinded input.
  """
  responding_address: Optional[Address]

  def __init__(self,
               data: Dict[str, Any],
               datasets: Dict[str, List[Dataset]] = {},
               paths: Optional[Dict[str, Directory]] = None,
               ) -> None:
    self.__data = data
    self.datasets = datasets
    self.paths = paths
    self.responding_address = None

  def __getitem__(self, key: str):
    if self.__data is None:
      raise KeyError(key)
    return self.__data[key]

  def __contains__(self, key):
    return key in self.__data


class PatientNode(ImageTreeInterface):
  """This is an ImageTree node, where each subnode contains related images.
  When filled can run through the pipelines process function.
  The Node contains AbstractInputs and only AbstractInput

  This class is responsible for:
    - Determine if there's sufficient data to run the pipeline processing
    - Creating InputContainer
  """
  #In most cases, each subnode is also a leaf.

  @dataclass
  class Options:
    InputContainerType: Type[InputContainer] = InputContainer
    "Type that this node should return from a extract_input_container call"

    container_path: Optional[Directory] = None
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

    ae_title: Optional[str] = None

    lazy: bool = False

    logger: Optional[Logger] = None

    input_container_type: Type[InputContainer] = InputContainer

  def __init__(self,
               args: Dict[str, Type[AbstractInput]],
               options: Options = Options(),
    ) -> None:
    super().__init__()
    self.options = options
    self.creation_time = datetime.now()
    self.study_date = None

    for arg_name, dicomnode_input_type in args.items():
      input_path: Optional[Directory] = None
      if self.options.container_path is not None:
        input_path = Directory(self.options.container_path / arg_name)

      input_options = self._get_input_options(dicomnode_input=dicomnode_input_type,
                                              input_path=input_path)
      self.data[arg_name] = dicomnode_input_type(options=input_options)

    # logger
    if self.options.logger is not None:
      self.logger = self.options.logger
    else:
      self.logger = getLogger(DICOMNODE_LOGGER_NAME)

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
      shutil.rmtree(self.options.container_path.path)
    return images_removed

  def validate_inputs(self) -> bool:
    valid = True
    for dicomnode_input in self.data.values():
      if isinstance(dicomnode_input, AbstractInput):
        input_container_valid = dicomnode_input.validate()
        self.logger.debug(f"Input {dicomnode_input.__class__.__name__} with Images {dicomnode_input.images} validates to: {input_container_valid}")
        valid &= input_container_valid
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
    series = {}

    path_directory: Optional[Dict[str, Directory]] = None
    if self.options.container_path is not None:
      path_directory = {}

    for arg_name, dicomnode_input in self.data.items():
      if isinstance(dicomnode_input, AbstractInput):
        self.logger.debug(f"Extracting input from {dicomnode_input.__class__.__name__}")
        data_directory[arg_name] = dicomnode_input.get_data()
        series[arg_name] = dicomnode_input.get_datasets()
        if path_directory is not None and dicomnode_input.container is not None:
          path_directory[arg_name] = dicomnode_input.container
      else:
        raise InvalidTreeNode # pragma: no cover

    self.logger.debug("Extracted data from all inputs")

    input_container = self.options.input_container_type(data_directory,
                                                        series,
                                                        path_directory)

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
      self.logger.info("dataset was rejected from all inputs")
      raise InvalidDataset()
    if self.study_date is None and 'StudyDate' in dicom:
      self._patch_study_date_for_inputs(dicom.StudyDate)

    self.images += added # This is not true, because of replacement
    return added

  def _get_input_options(self, dicomnode_input: Type[AbstractInput], input_path: Optional[Directory]):
    return dicomnode_input.Options(
        ae_title=self.options.ae_title,
        data_directory = input_path,
        logger=self.options.logger,
        lazy=self.options.lazy
      )

  def _patch_study_date_for_inputs(self, patch_date: str):
    if self.study_date is not None:
      return

    self.study_date = patch_date

    for dicomnode_input in self.data.values():
      if isinstance(dicomnode_input, AbstractInput):
        dicomnode_input._study_date = patch_date
      else:
        raise InvalidTreeNode()

  @property
  def stored_images(self) -> int:
    images = 0
    for dicomnode_input in self.data.values():
      if isinstance(dicomnode_input, AbstractInput):
        images += dicomnode_input.images

    return images

  def __str__(self):
    sub_strings = [f"PatientNode created at {self.creation_time}"] + [f"  {input_}" for input_ in self.data.values()]
    return "\n".join(sub_strings)



class PipelineTree(ImageTreeInterface):
  """A more specialized ImageTree, which is used by a dicom node to keep track
  of studies.

  This is the root node of the servers ImageTree,
  It contains leafs of PatientNodes and only PatientNodes.
  Most function will raise the InvalidTreeNode exception if this is violated.

  The main responsibility of this class is to manage Patient Nodes: So
    - Creating new PatientNodes when needed
    - Deleting PatientNodes, when they're expired or processed

  The pipeline tree is thread safe with locks
  """

  @dataclass
  class Options:
    """Options for a Pipeline Tree"""

    # This class mainly exists to have it's options injected to it from on high
    # rather that having all of these thing parse into it from args
    ae_title: Optional[str] = None
    "AE title of node"

    data_directory: Optional[Directory] = None
    "Root directory for file storage"

    lazy: bool = False
    "If underlying inputs should use lazy datasets"

    logger: Optional[Logger] = None
    "Logger to send message to"

    input_container_type: Type[InputContainer] = InputContainer
    "type of input containers get_patient_input_container should return"

    patient_container: Type[PatientNode] = PatientNode
    "Type of node that's under this tree."



  def __init__(self,
               patient_identifier: int,
               pipelineArgs: Dict[str, Type[AbstractInput]],
               options = Options()
    ) -> None:
    """A Dicom tree of height 2* where the first node is the patient identifier
    The second layer is instances of the abstract input.

    Args:
        patient_identifier (int): The Dicom tag for to separate dicom series
        pipelineArgs (Dict[str, Type]): A "Blueprint" for each node
        Options: PipelineTree.Options: Options for the pipeline tree

    Raises:
        InvalidRootDataDirectory: _description_
    """
    # ImageTreeInterface required attributes
    super().__init__()

    # Args setup
    self.patient_identifier_tag: int = patient_identifier # Move this to options?
    self.tree_node_definition: Dict[str, Type[AbstractInput]] = pipelineArgs
    self.options = options

    #Logger Setup
    if self.options.logger is None:
      self.logger = getLogger(DICOMNODE_LOGGER_NAME)
    else:
      self.logger = self.options.logger

    #Load File state
    if self.options.data_directory is None:
      # There are no files to load if it's in memory
      return

    for patient_directory in self.options.data_directory:
      options = self._get_patient_container_options(Directory(patient_directory))

      self[patient_directory.name] = PatientNode(self.tree_node_definition, options)

  def add_image(self, dicom : Dataset) -> int:
    key = self.get_patient_id(dicom)

    if key not in self:
      input_container_path: Optional[Directory] = None
      if self.options.data_directory is not None:
        input_container_path = Directory(self.options.data_directory / key)

      options = self._get_patient_container_options(input_container_path)
      self[key] = PatientNode(self.tree_node_definition, options)

    patient_node = self[key]
    if isinstance(patient_node, PatientNode):
      added = patient_node.add_image(dicom)
      self.images += added
      return added

    raise InvalidTreeNode # pragma: no cover

  def get_storage_directory(self, identifier: Any) -> Optional[Directory]:
    if self.options.data_directory is None:
      return None

    if isinstance(identifier, DicomSeries):
      val = identifier[self.patient_identifier_tag]
      if val is not None and not isinstance(val, List):
        return Directory(self.options.data_directory / str(val.value))
      else:
        return None

    if isinstance(identifier, Dataset):
      return Directory(self.options.data_directory / str(identifier[self.patient_identifier_tag].value))

    if isinstance(identifier, str):
      return Directory(self.options.data_directory / identifier)

    raise TypeError("Unknown identifier type")

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
    if patient_id not in self:
      return False
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
      self.logger.error(f"{hex(self.patient_identifier_tag)} not in dataset")
      self.logger.error("Patient Identifier tag not in dataset")
      raise InvalidDataset()

    value = dataset[self.patient_identifier_tag].value

    if value is None:
      self.logger.error(f"Input dataset have tag {hex(self.patient_identifier_tag)}\
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
      else: #pragma: no cover
        raise InvalidTreeNode
    if len(to_be_removed) > 0:
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

    old_size = len(self.data)

    for patient_id, patient_node in self.data.items():
      if patient_id in patient_ids:
        if isinstance(patient_node, PatientNode):
          removed_images += patient_node.clean_up()
        else:
          raise InvalidTreeNode # pragma: no cover
      else:
        new_data_dict[patient_id] = patient_node

    self.images -= removed_images
    self.data = new_data_dict

    new_size = len(self.data)
    self.logger.debug(f"Removed {removed_images} images from {old_size - new_size} Patients")

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

    self.logger.debug(f"Removed {patient_id} and {removed_images} images from Pipeline")

  def _get_patient_container_options(self, container_path: Optional[Directory]) -> PatientNode.Options:
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
        logger=self.logger,
        lazy=self.options.lazy,
        input_container_type=self.options.input_container_type,
      )

  def __str__(self) -> str:
    sub_strings = [f"Pipeline Tree - {len(self.data)} Patients - {self.images} images total"]
    for id_, patient_node in self.data.items():
      patient_node_lines = str(patient_node).split("\n")
      lines_to_be_added = []
      for i, line in enumerate(patient_node_lines):
        if i == 0:
          # This is Abstract Input
          lines_to_be_added.append(f"  {id_} - {line}")
        else:
          # This is string representing an Abstract Input
          lines_to_be_added.append(f"    {line}")
      sub_strings += lines_to_be_added

    return "\n".join(sub_strings)

__all__ = [
  'InputContainer',
  'PatientNode',
  'PipelineTree'
]