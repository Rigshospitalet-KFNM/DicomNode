# Python Standard Library
from time import sleep
from threading import Event as ThreadingEvent # normal Events refer to pynetdicom's events
from typing import Dict, List, Optional

# Third party packages
from pydicom import Dataset
from pynetdicom.ae import ApplicationEntity
from pynetdicom.presentation import AllStoragePresentationContexts, build_context
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind, StudyRootQueryRetrieveInformationModelMove, PatientRootQueryRetrieveInformationModelFind, PatientRootQueryRetrieveInformationModelMove  #type: ignore
from pynetdicom import events
# Dicomnode packages

MOVE_ENDPOINT = 10000
ENDPOINT_PORT = 50000
ENDPOINT_AE_TITLE = "ENDPOINT_AE"

PATIENT_ID = 0x0010_0020

class TestStorageEndpoint():
  """This class is a AE wrapper, for getting and extracting data

  It's a more sophisticated version of the get_test_ae
  """
  def __init__(self,
               endpoint_port=ENDPOINT_PORT,
               endpoint_ae_title=ENDPOINT_AE_TITLE,
               release_event: Optional[ThreadingEvent] = None,
               move_endpoint = MOVE_ENDPOINT,
               ) -> None:
    self._storage = {}
    self.ae_title = endpoint_ae_title
    self.endpoint_port = endpoint_port
    self.running = False
    self.accepted_associations = 0
    self.ae = ApplicationEntity(endpoint_ae_title)
    self.ae.supported_contexts = AllStoragePresentationContexts
    self.ae.add_supported_context(StudyRootQueryRetrieveInformationModelFind)
    self.ae.add_supported_context(StudyRootQueryRetrieveInformationModelMove)
    self.ae.add_supported_context(PatientRootQueryRetrieveInformationModelMove)
    self.ae.add_supported_context(PatientRootQueryRetrieveInformationModelFind)
    self.release_event = release_event
    self.move_target = ('127.0.0.1', move_endpoint)


  @property
  def storage(self) -> Dict[str, List[Dataset]]:
    return self._storage

  def handle_C_store(self, evt: events.Event):
    dataset = evt.dataset
    dataset.file_meta = evt.file_meta

    if PATIENT_ID not in dataset:
      print("Patient ID not in dataset")
      return 0xC200

    if dataset[PATIENT_ID].value in self.storage:
      self.storage[dataset[PATIENT_ID].value].append(dataset)
    else:
      self.storage[dataset[PATIENT_ID].value] = [dataset]

    return 0x0000

  def handle_C_move(self, evt):
    # Yield destination
    yield self.move_target
    # Yield Number of operations
    yield 0
    # Yield datasets, but there's none!

  def handle_C_find(self, evt):
    return

  def _accepted(self, evt: events.Event):
    self.accepted_associations += 1

  def _released(self, evt: events.Event):
    if self.release_event is not None:
      self.release_event.set()

  def open(self):
    if not self.running:
      self.ae.start_server(("127.0.0.1", self.endpoint_port),
                           evt_handlers=[
                             (events.EVT_RELEASED, self._released),
                             (events.EVT_ACCEPTED, self._accepted),
                             (events.EVT_C_STORE, self.handle_C_store),
                             (events.EVT_C_FIND, self.handle_C_find),
                             (events.EVT_C_MOVE, self.handle_C_move),
                           ],
                           block=False,
                           ae_title=self.ae_title
                           )
      self.running = True

  def close(self):
    if self.running:
      self.ae.shutdown()
      self.running = False

  def wait_till_ready(self):
    if self.running:
      while self.ae.active_associations != []:
        sleep(0.005)
    return