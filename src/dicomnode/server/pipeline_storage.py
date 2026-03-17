__author__ = "Demiguard"

# Python Standard Library
from datetime import datetime
from threading import Lock, get_native_id

from typing import Dict, Iterable, List, Tuple, Type

# Third Party Python Packages
from pydicom import Dataset

# Dicomnode Library Packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.dicom import DicomIdentifier
from dicomnode.dicom.series import DicomSeries
from dicomnode.data_structures.image_tree import ImageTreeInterface
from dicomnode.dicom.dimse import Address
from dicomnode.lib.io import Directory
from dicomnode.lib.exceptions import InvalidDataset, InvalidRootDataDirectory,\
                                      InvalidTreeNode
from dicomnode.data_structures.defaulting_dict import DefaultingDict
from dicomnode.config import DicomnodeConfig
from dicomnode.server.patient_node import PatientNode
from dicomnode.server.input import AbstractInput


def create_set():
  return set()

def create_list():
  return list()

class PipelineStorage:
  """Provides an interface to store and extract abstract inputs thread safely

  To do so it must follow the following rules

  1: Before adding an dataset, the thread must register that it have added
  2: Before extracting no other threads must be adding

  """


  def __init__(self,
               node_structure: Dict[str, Type[AbstractInput]],
               identifier: DicomIdentifier,
               config: DicomnodeConfig
               ) -> None:

    self.identifier = identifier
    self.node_structure = node_structure
    self.config = config

    def create_patient_node():
      return PatientNode(node_structure, config)

    self.storage: DefaultingDict[str, PatientNode] = DefaultingDict(create_patient_node)
    self.thread_registration: DefaultingDict[str, set[int]] = DefaultingDict(create_set)
    self.thread_additions: DefaultingDict[int, set[str]] = DefaultingDict(create_set)
    self.failed_additions: DefaultingDict[int, List[Dataset]] = DefaultingDict(create_list)

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

  def add_images(self, datasets: Iterable[Dataset]):
    for dataset in datasets:
      self.add_image(dataset)

  def extract_input_container(self, thread_id = None):
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
          extracted_input_containers.append((dicom_identifier,self.storage.extract(dicom_identifier)))
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
