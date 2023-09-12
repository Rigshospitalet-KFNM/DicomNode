"""Dataclasses for pynetdicom.event extraction.

These function extracts relevant information from the pynetdicom.event.Event
For futher processing
"""


# Python Standard Library
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Third party packages
from pydicom import Dataset
from pynetdicom.events import Event, EventType, EVT_ACCEPTED, EVT_RELEASED, EVT_C_STORE

# Dicomnode packages


class AssociationTypes(Enum):
  StoreAssociation = 1

class AssocationContainer:
  pass

##### Dataclasses ######
@dataclass
class AcceptedContainer(AssocationContainer):
  assocation_id : int # Connects events triggered by this assocation
  assocation_types : set[AssociationTypes] # Checks if this assocation is a store association or not
  assocation_ae : str
  assocation_ip : Optional[str]


@dataclass
class ReleasedContainer(AssocationContainer):
  assocation_id : int # Connects events triggered by this assocation
  assocation_types : set[AssociationTypes] # Checks if this assocation is a store association or not
  assocation_ae_title : str
  assocation_ip : Optional[str]


@dataclass
class CStoreContainer(AssocationContainer):
  assocation_id : int
  dataset: Dataset

##### Corosponding Factory #####
class AssociationContainerFactory:
  def __get_event_id(self, event: Event) -> int:
    if event.assoc.native_id is None:
      raise ValueError("") # pragma: no cover
    return event.assoc.native_id

  def build_assocation_accepted(self, event: Event) -> AcceptedContainer:
    association_types = set()
    for requested_context in event.assoc.requestor.requested_contexts:
      if requested_context.abstract_syntax is not None \
          and requested_context.abstract_syntax.startswith("1.2.840.10008.5.1.4.1.1"):
        association_types.add(AssociationTypes.StoreAssociation)
    assocation_ae = event.assoc.requestor.ae_title
    assocation_ip = event.assoc.requestor.address

    return AcceptedContainer(
      self.__get_event_id(event),
      association_types,
      assocation_ae,
      assocation_ip,
    )


  def build_assocation_released(self, event: Event) -> ReleasedContainer:
    association_types = set()
    for requested_context in event.assoc.requestor.requested_contexts:
      if requested_context.abstract_syntax is not None \
          and requested_context.abstract_syntax.startswith("1.2.840.10008.5.1.4.1.1"):
        association_types.add(AssociationTypes.StoreAssociation)
    assocation_ae = event.assoc.requestor.ae_title
    assocation_ip = event.assoc.requestor.address

    return ReleasedContainer(
      self.__get_event_id(event),
      association_types,
      assocation_ae,
      assocation_ip,
      )


  def build_assocation_c_store(self, event: Event) -> CStoreContainer:
    dataset = event.dataset
    dataset.file_meta = event.file_meta

    return CStoreContainer(self.__get_event_id(event), dataset)


  # I'm internal debate over cutting this function, since there's currently
  # No caller to this function
  def from_event(self, event: Event) -> AssocationContainer:
    """Convert an event of any type to the related AssocationContainer
    """
    if event.event == EVT_ACCEPTED:
      return self.build_assocation_accepted(event)
    elif event.event == EVT_RELEASED:
      return self.build_assocation_released(event)
    elif event.event == EVT_C_STORE:
      return self.build_assocation_c_store(event)
    else:
      raise Exception
