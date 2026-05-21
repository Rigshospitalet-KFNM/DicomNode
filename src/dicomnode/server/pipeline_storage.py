__author__ = "Demiguard"

# Python Standard Library
from abc import ABC, abstractmethod
from datetime import datetime
from threading import Lock, get_native_id, enumerate as thread_enumeration, get_ident
from typing import Dict, Iterable, List, Optional, Tuple, Type

# Third Party Python Packages
from pynetdicom.association import Association
from pydicom import Dataset

# Dicomnode Library Packages
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.lib.logging import get_logger
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
  def reset_allocation(self) -> None:
    """If the pipeline have any auxiliary state, then this function resets that"""
    raise NotImplemented

  @abstractmethod
  def add_image(self, dicom_dataset: Dataset, thread_id = None):
    raise NotImplemented

  def add_images(self, datasets: Iterable[Dataset]):
    for dataset in datasets:
      self.add_image(dataset)

  @abstractmethod
  def extract_input_container(self, thread_id = None) -> Tuple[List[Tuple[str, PatientNode]], List[Dataset]]:
    raise NotImplemented

  @abstractmethod
  def remove_expired_studies(self, expiry_time: datetime):
    raise NotImplemented

class _HeartBeat(ABC):
  """This is the interface class, that the reactive something that have added to
  the reactive pipeline.
  """
  @abstractmethod
  def __init__(self, identifier) -> None:
    raise NotImplemented

  @abstractmethod
  def is_active(self) -> bool:
    raise NotImplemented

  @abstractmethod
  def __eq__(self, value: object) -> bool:
    raise NotImplemented

  @abstractmethod
  def __hash__(self) -> int:
    raise NotImplemented

  def __str__(self) -> str:
    return f"Heartbeat - alive: {self.is_active()}"

  def __repr__(self) -> str:
    return str(self)

class _ThreadHeartBeat(_HeartBeat):
  def __init__(self, identifier: int) -> None:
    self.identifier = identifier

  def is_active(self) -> bool:
    threads = [thread.ident for thread in thread_enumeration() if thread.ident is not None]

    return self.identifier in threads

  def __eq__(self, value: object) -> bool:
    if isinstance(value, _ThreadHeartBeat):
      return value.identifier == self.identifier

    return False

  def __hash__(self) -> int:
    return hash(self.identifier)


class _AssocHeartBeat(_HeartBeat):
  def __init__(self, identifier: Association) -> None:
    self.assoc = identifier

  def is_active(self) -> bool:
    return self.assoc.is_alive()

  def __eq__(self, value: object) -> bool:
    if isinstance(value, _AssocHeartBeat):
      return self.assoc.ident == value.assoc.ident

    return False

  def __hash__(self) -> int:
    return hash(self.assoc.ident)

def _make_heartbeat(identifier) -> _HeartBeat:
  if identifier is None:
    return _ThreadHeartBeat(get_ident())
  if isinstance(identifier, int):
    return _ThreadHeartBeat(identifier)
  if isinstance(identifier, Association):
    return _AssocHeartBeat(identifier)

  raise TypeError(f"Could not construct a heart beat from {identifier}")


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
    self.thread_registration: DefaultingDict[str, set[_HeartBeat]] = DefaultingDict(create_set)
    """A patient id mapping that stores which threads have added to each patient"""

    self.heartbeats_additions: DefaultingDict[_HeartBeat, set[str]] = DefaultingDict(create_set)
    """A mapping that holds which """

    self.failed_additions: DefaultingDict[_HeartBeat, List[Dataset]] = DefaultingDict(create_list)
    """"""

    self.master_lock = Lock()

  def reset_allocation(self):
    with self.master_lock:
      self.heartbeats_additions = DefaultingDict(create_set)
      self.thread_registration = DefaultingDict(create_set)
      self.failed_additions = DefaultingDict(create_list)

  def add_image(self, dicom_dataset: Dataset, thread_id=None):
    dicom_identifier = self.identifier(dicom_dataset)

    heartbeat = _make_heartbeat(thread_id)

    with self.master_lock:
      self.thread_registration[dicom_identifier].add(heartbeat)

      try:
        self.storage[dicom_identifier].add_dataset(dicom_dataset)
        self.heartbeats_additions[heartbeat].add(dicom_identifier)
      except InvalidDataset:
        self.failed_additions[heartbeat].append(dicom_dataset)

    """I kinda want to make a small little programmatic comment here. Ideally
       you would have made a `with` statement because it's kind of a mess if a
       thread dies or doesn't unregister itself, then It
  """

  def extract_input_container(self, thread_id = None):
    logger = get_logger()
    heartbeat = _make_heartbeat(thread_id)
    extracted_input_containers: List[Tuple[str,PatientNode]] = []

    with self.master_lock:
      for dicom_identifier in self.heartbeats_additions[heartbeat]:
        thread_set = self.thread_registration[dicom_identifier]
        thread_set.discard(heartbeat)

        should_extract = True

        for heartbeat_ in thread_set:
          should_extract &= not heartbeat_.is_active()

        if not should_extract:
          logger.info(f"Thread {heartbeat} is not extracting {dicom_identifier} because there's {len(thread_set)} other threads active.")
          continue

        if self.storage[dicom_identifier].validate():
          extracted_input_containers.append((dicom_identifier, self.storage.extract(dicom_identifier)))
          del self.thread_registration[dicom_identifier]

    del self.heartbeats_additions[heartbeat]
    failed_datasets = self.failed_additions.extract(heartbeat)

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

  def reset_allocation(self) -> None:
    """"""
    return None

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
