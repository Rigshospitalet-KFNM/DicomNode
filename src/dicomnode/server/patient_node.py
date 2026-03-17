""""""

__author__ = "Demiguard"

# Python standard library
from datetime import datetime
from functools import reduce
import logging
from typing import Dict, Optional, Type

# Third party modules
from pydicom import Dataset

# Dicomnode Modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.config import DicomnodeConfig
from dicomnode.server.input_container import InputContainer
from dicomnode.server.input import AbstractInput



class PatientNode:
  def __init__(
      self,
      node_structure: Dict[str, Type[AbstractInput]],
      config: DicomnodeConfig
    ) -> None:
    self._nodes: Dict[str, AbstractInput] = { key : InputType(config) for key, InputType in node_structure.items()}
    self._created = datetime.now()
    self._config = config
    self._study_date: Optional[str] = None

  def __iter__(self):
    for node in self._nodes.values():
      yield node

  def __len__(self):
    return reduce(lambda x,y : x + y, [len(node) for node in self], 0)

  def add_dataset(self, dataset: Dataset):
    added_dataset = False
    for node in self:
      try:
        node.add_image(dataset)
        added_dataset = True
      except InvalidDataset:
        pass

    if not added_dataset:
      raise InvalidDataset

    self.study_date = dataset.StudyDate if 'StudyDate' in dataset else None


  @property
  def study_date(self):
    return self._study_date

  @study_date.setter
  def study_date(self, value: Optional[str]):
    if value is None:
      return

    if self._study_date is not None:
      return

    self._study_date = value
    for node in self:
      node.study_date = value


  def validate(self) -> bool:
    logger = logging.getLogger(DICOMNODE_LOGGER_NAME)
    logger.info(f"Validating: {self}")
    return all(node.validate() for node in self)

  def is_expired(self, expiry_time: datetime) -> bool:
    return self._created < expiry_time

  def items(self):
    return self._nodes.items()

  def grind(self) -> InputContainer:
    return InputContainer(
      data={
        key : input_.grind() for key, input_ in self.items()
      },
      datasets={
        key : input_.get_datasets() for key, input_ in self.items()
      },
      paths={
        key : input_.container for key, input_ in self.items() if input_.container is not None
      }
    )

  def clean_up(self):
    for node in self:
      node.clean_up()

  def __str__(self) -> str:
    base = f"Patient Node with {len(self)} images:\n"
    for node in self:
      base += f"  {node}\n"

    return base

  def __repr__(self) -> str:
    return str(self)
