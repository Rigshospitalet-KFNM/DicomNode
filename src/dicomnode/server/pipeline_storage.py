__author__ = "Demiguard"

# Python Standard Library
from abc import ABC, abstractmethod
from datetime import datetime
from threading import Lock, get_native_id
from typing import Dict, Iterable, List, Optional, Tuple, Type

# Third Party Python Packages
from pydicom import Dataset

# Dicomnode Library Packages
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.data_structures.defaulting_dict import DefaultingDict
from dicomnode.config import DicomnodeConfig
from dicomnode.server.patient_node import PatientNode
from dicomnode.server.input import AbstractInput


def create_set():
  return set()

def create_list():
  return list()

class PipelineStorage(ABC):
  @abstractmethod
  def __init__(self,
               node_structure: Dict[str, Type[AbstractInput]],
               config: DicomnodeConfig
               ) -> None:
    raise NotImplemented

  @abstractmethod
  def add_image(self, dicom_dataset: Dataset, thread_id: Optional[int] = None):
    raise NotImplemented

  def add_images(self, datasets: Iterable[Dataset]):
    for dataset in datasets:
      self.add_image(dataset)

  @abstractmethod
  def extract_input_container(self, thread_id: Optional[int] = None) -> Tuple[List[Tuple[str, PatientNode]], List[Dataset]]:
    raise NotImplemented

  @abstractmethod
  def remove_expired_studies(self, expiry_time: datetime):
    raise NotImplemented

class ReactivePipelineStorage(PipelineStorage):
  """Provides an interface to store and extract abstract inputs thread safely

  To do so it must follow the following rules

  1: Before adding an dataset, the thread must register that it have added
  2: Before extracting no other threads must be adding

  """


  def __init__(self,
               node_structure: Dict[str, Type[AbstractInput]],
               config: DicomnodeConfig
               ) -> None:

    self.identifier = config.IDENTIFIER
    self.node_structure = node_structure
    self.config = config

    def create_patient_node(key):
      return PatientNode(key, node_structure, config)

    self.storage: DefaultingDict[str, PatientNode] = DefaultingDict(create_patient_node)
    """The Storage where datasets are stored until they are ready to be processed"""
    self.thread_registration: DefaultingDict[str, set[int]] = DefaultingDict(create_set)
    """A patient id mapping that stores which threads have added to each patient"""

    self.thread_additions: DefaultingDict[int, set[str]] = DefaultingDict(create_set)
    """A mapping that holds which """

    self.failed_additions: DefaultingDict[int, List[Dataset]] = DefaultingDict(create_list)
    """"""

    self.master_lock = Lock()


  def add_image(self, dicom_dataset: Dataset, thread_id=None):
    dicom_identifier = self.identifier(dicom_dataset)

    if thread_id is None:
      thread_id = get_native_id()

    with self.master_lock:
      self.thread_registration[dicom_identifier].add(thread_id)

      try:
        self.storage[dicom_identifier].add_dataset(dicom_dataset)
        self.thread_additions[thread_id].add(dicom_identifier)
      except InvalidDataset:
        self.failed_additions[thread_id].append(dicom_dataset)

    """I kinda want to make a small little programmatic comment here. Ideally
       you would have made a `with` statement because it's kind of a mess if a
       thread dies or doesn't unregister itself, then It
  """


  def extract_input_container(self, thread_id: Optional[int] = None):
    if thread_id is None:
      thread_id = get_native_id()

    extracted_input_containers: List[Tuple[str,PatientNode]] = []

    with self.master_lock:
      for dicom_identifier in self.thread_additions[thread_id]:
        thread_set = self.thread_registration[dicom_identifier]
        thread_set.discard(thread_id)

        if 0 < len(thread_set):
          continue

        if self.storage[dicom_identifier].validate():
          extracted_input_containers.append((dicom_identifier, self.storage.extract(dicom_identifier)))
          del self.thread_registration[dicom_identifier]

    del self.thread_additions[thread_id]
    failed_datasets = self.failed_additions.extract(thread_id)

    return extracted_input_containers, failed_datasets

  def remove_expired_studies(self, expiry_time : datetime):
    """Removes any PatientNode in the tree that have expired.

    Args:
      expiry_time (datetime): Any study created before expiry_time is considered to be expired.
    """
    dirty_identifiers = []

    for identifier, node in self.storage:
      if node.is_expired(expiry_time):
        dirty_identifiers.append(identifier)

    for identifier in dirty_identifiers:
      del self.storage[identifier]

  def __str__(self) -> str:
    base = f"Pipeline Storage with {len(self.storage)} patients\n"
    for patient_id, node in self.storage:
      lines = str(node).split('\n')


    return base

class PassivePipelineStorage(PipelineStorage):
  """A PipelineStorage, that is ignorant of the incoming threads. It's not
  thread safe, in that and relies 100 % on correct implementation of the
  underlying abstract Inputs.
  - Just so you are aware some SCU's will send things over multiple associations
  so you cannot rely on that.
  """
  def __init__(self, node_structure: Dict[str, Type[AbstractInput]], config: DicomnodeConfig) -> None:
    self.node_structure = node_structure
    self.config = config

    def create_patient_node(key):
      return PatientNode(key, node_structure, config)

    self.tree: DefaultingDict[str, PatientNode] = DefaultingDict(create_patient_node)
    self.failed_datasets = []

  def add_image(self, dicom_dataset, thread_id=None):
    identifier = self.config.IDENTIFIER(dicom_dataset)

    node = self.tree[identifier]

    try:
      node.add_dataset(dicom_dataset)
    except InvalidDataset:
      self.failed_datasets.append(dicom_dataset)

  def extract_input_container(self, thread_id: int | None = None) -> Tuple[List[Tuple[str, PatientNode]], List[Dataset]]:
    failed_datasets = self.failed_datasets
    self.failed_datasets = []

    node_to_process = []

    for patient_id, node in self.tree:
      if node.validate():
        node_to_process.append((patient_id, node))

    for patient_id, _ in node_to_process:
      del self.tree[patient_id]

    return node_to_process, failed_datasets

  def remove_expired_studies(self, expiry_time: datetime):
    dirty_identifiers = []

    for identifier, node in self.tree:
      if node.is_expired(expiry_time):
        dirty_identifiers.append(identifier)

    for identifier in dirty_identifiers:
      del self.tree[identifier]
