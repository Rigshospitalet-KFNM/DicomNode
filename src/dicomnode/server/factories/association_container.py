"""Dataclasses for pynetdicom.event extraction.

These function extracts relevant information from the pynetdicom.event.Event
For further processing
"""


# Python Standard Library
from dataclasses import dataclass
from enum import Enum
from threading import get_native_id
from typing import Optional

# Third party packages
from pydicom import Dataset
from pynetdicom.events import Event

# Dicomnode packages

class AssociationTypes(Enum):
  """The different types of associations that a dicom node can support

  """
  STORE_ASSOCIATION = 1

@dataclass
class AssociationContainer: # FUCK THIS IS BAD NAMING
  """Base Class for AssociationContainers these class extracts the information
  of a pynetdicom."""
  association_id : int
  """The id of the trigger association. Is the thread id of association"""
  association_ip : Optional[str]
  """ip address of the triggering association"""
  association_ae : str
  """AE title of the triggering association"""
  handling_thread : int
  """The Thread ID of the handling thread, used when the thread pass work to
  another thread, for reference back. Output from get_native_id
  """


##### Dataclasses ######
@dataclass
class AcceptedContainer(AssociationContainer):
  """Extracted data from an AssociationAccepted event."""
  association_types : set[AssociationTypes]
  """Checks if this association is a store association or not"""


@dataclass
class ReleasedContainer(AssociationContainer):
  """Extracted data from an AssociationReleased event."""
  association_types : set[AssociationTypes]
  """Checks if this association is a store association or not"""


@dataclass
class CStoreContainer(AssociationContainer):
  """Extracted data from a evt.EVT_C_STORE"""
  dataset: Dataset

##### Corresponding Factory #####
class AssociationContainerFactory:
  """Factory class for extracting data from various pynetdicom.Event.

  The idea is that if pynetdicom changes their API, then it's this part that'll
  break, rather than all over the node
  """
  # I'm really not sure this is worth the additional abstraction, but perfect code
  # doesn't exists!
  def __get_event_id(self, event: Event) -> int:
    if event.assoc.native_id is None:
      # tbh I think you could just threading.get_native_id
      raise ValueError("") # pragma: no cover
    return event.assoc.native_id

  def build_association_accepted(self, event: Event) -> AcceptedContainer:
    """Extracts information from an accepted event and puts into a accepted
    container.

    Args:
        event (Event): An evt.EVT_ACCEPTED event

    Returns:
        AcceptedContainer: dicomnode version of An evt.EVT_ACCEPTED event
    """
    association_types = set()
    for requested_context in event.assoc.requestor.requested_contexts:
      if requested_context.abstract_syntax is not None \
          and requested_context.abstract_syntax.startswith("1.2.840.10008.5.1.4.1.1"):
        association_types.add(AssociationTypes.STORE_ASSOCIATION)
    association_ae = event.assoc.requestor.ae_title
    association_ip = event.assoc.requestor.address

    return AcceptedContainer(
      self.__get_event_id(event),
      association_ip,
      association_ae,
      get_native_id(),
      association_types,
    )


  def build_association_released(self, event: Event) -> ReleasedContainer:
    """Extracts information from an release event and puts into a release
    container.

    Args:
        event (Event): An evt.EVT_RELEASED event

    Returns:
        ReleasedContainer: dicomnode version of An evt.EVT_RELEASED event
    """
    association_types = set()
    for requested_context in event.assoc.requestor.requested_contexts:
      if requested_context.abstract_syntax is not None \
          and requested_context.abstract_syntax.startswith("1.2.840.10008.5.1.4.1.1"):
        association_types.add(AssociationTypes.STORE_ASSOCIATION)
    association_ae = event.assoc.requestor.ae_title
    association_ip = event.assoc.requestor.address

    return ReleasedContainer(
      self.__get_event_id(event),
      association_ip,
      association_ae,
      get_native_id(),
      association_types,
    )


  def build_association_c_store(self, event: Event) -> CStoreContainer:
    """Extracts information from an C-Store event and puts into a C-Store
    container.

    Args:
        event (Event): An evt.EVT_C_STORE event

    Returns:
        CStoreContainer: dicomnode version of An evt.EVT_C_STORE event
    """
    dataset = event.dataset
    dataset.file_meta = event.file_meta

    return CStoreContainer(self.__get_event_id(event),
                           event.assoc.requestor.address,
                           event.assoc.requestor.ae_title,
                           get_native_id(),
                           dataset)
