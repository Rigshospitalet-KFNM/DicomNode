"""This module is the base class for all pipelines

  The life cycle of the pipeline is:

 - Storage Association is created
 - A number of C-stores is send to the pipeline.
 - The Association is released
 - For each updated patientTree, checks if all InputDataClasses validates
 - For each patient where they do, process them and dispatch them

"""

__author__ = "Christoffer Vilstrup Jensen"

# Standard lib
from copy import deepcopy
import logging
from logging import getLogger
from os import chdir, getcwd
from pathlib import Path
from queue import Queue, Empty
import shutil
from sys import stdout
from threading import Thread
from typing import Any, Dict, List, NoReturn, Optional, Set, TextIO, Type, Union

# Third part packages
from pynetdicom import evt
from pynetdicom.ae import ApplicationEntity as AE
from pynetdicom.presentation import AllStoragePresentationContexts, PresentationContext, VerificationPresentationContexts
from pydicom import Dataset

# Dicomnode packages
from dicomnode.lib.dicom_factory import Blueprint, DicomFactory, FillingStrategy
from dicomnode.lib.dimse import Address
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured
from dicomnode.lib.io import TemporaryWorkingDirectory
from dicomnode.lib.logging import log_traceback, set_logger
from dicomnode.server.assocation_container import AcceptedContainer, AssociationContainerFactory, AssociationTypes, CStoreContainer, ReleasedContainer
from dicomnode.server.input import AbstractInput
from dicomnode.server.pipeline_tree import PipelineTree, InputContainer, PatientNode
from dicomnode.server.maintenance import MaintenanceThread
from dicomnode.server.output import PipelineOutput, NoOutput

class AbstractPipeline():
  """Base Class for an image processing Pipeline
  Creates a SCP server and starts it, unless start=False is passed

  Should be subclassed with your implementation of the process function at least.

  Check tutorials/ConfigurationOverview.md for an overview of attributes
  """

  # Directory for file Processing
  processing_directory: Optional[Path] = None

  # Maintenance Configuration
  maintenance_thread: Type[MaintenanceThread] = MaintenanceThread
  study_expiration_days: int = 14

  # Input configuration
  input: Dict[str, Type[AbstractInput]] = {}
  input_config: Dict[str, Dict[str, Any]] = {}
  patient_identifier_tag: int = 0x00100020 # Patient ID
  data_directory: Optional[Path] = None
  lazy_storage: bool = False
  pipelineTreeType: Type[PipelineTree] = PipelineTree
  PatientContainerType: Type[PatientNode] = PatientNode
  InputContainerType: Type[InputContainer] = InputContainer

  #DicomGeneration
  dicom_factory: Optional[DicomFactory] = None
  filling_strategy: FillingStrategy = FillingStrategy.DISCARD
  header_blueprint: Optional[Blueprint] = None
  c_move_blueprint: Optional[Blueprint] = None

  # Output Configuration
  output: Type[PipelineOutput] = PipelineOutput

  # Dicom communication configuration tags
  ae_title: str = "Your_AE_TITLE"
  ip: str = 'localhost'
  port: int = 104
  supported_contexts: List[PresentationContext] = AllStoragePresentationContexts
  require_called_aet: bool = True
  require_calling_aet: List[str] = []
  known_endpoints: Dict[str, Address] = {}
  _assocations_responds_addresses: Dict[int, Address] = {}
  assocation_container_factory: Type[AssociationContainerFactory] = AssociationContainerFactory
  default_response_port: int = 104

  #Logging Configuration
  backup_weeks: int = 8
  log_date_format = "%Y/%m/%d %H:%M:%S"
  log_output: Optional[Union[TextIO, Path, str]] = stdout
  log_level: int = logging.INFO
  log_format: str = "%(asctime)s %(name)s %(levelname)s %(message)s"
  disable_pynetdicom_logger: bool = True

  def __init__(self) -> None:
    # This function starts and opens the server
    #
    # 1. Logging
    # 2. File system
    # 3. Create Server

    # logging
    if isinstance(self.log_output, str):
      self.log_output = Path(self.log_output)

    self.logger = set_logger(
      log_output=self.log_output,
      log_level=self.log_level,
      format=self.log_format,
      date_format=self.log_date_format,
      backupCount=self.backup_weeks
    )

    if self.disable_pynetdicom_logger:
      getLogger("pynetdicom").setLevel(logging.CRITICAL + 1)

    # Load any previous state
    if self.data_directory is not None:
      if not isinstance(self.data_directory, Path):
        self.data_directory = Path(self.data_directory)

      if self.data_directory.is_file():
        raise IncorrectlyConfigured("The root data directory exists as a file.")

      if not self.data_directory.exists():
        self.data_directory.mkdir(parents=True)

    pipeline_tree_options = self.pipelineTreeType.Options(
      ae_title=self.ae_title,
      data_directory=self.data_directory,
      factory=self.dicom_factory,
      filling_strategy=self.filling_strategy,
      header_blueprint=self.header_blueprint,
      lazy=self.lazy_storage,
      input_container_type=self.InputContainerType,
      patient_container=self.PatientContainerType,
    )

    self.data_state: PipelineTree = self.pipelineTreeType(
      self.patient_identifier_tag,
      self.input,
      pipeline_tree_options
    )

    self._maintenance_thread = self.maintenance_thread(
      self.data_state, self.study_expiration_days, daemon=True)

    self._assocation_container_factory = self.assocation_container_factory()

    # Server validations and creation.
    self.ae = AE(ae_title = self.ae_title)
    # You need VerificationPresentationContexts for ECHOSCU
    # and you want ECHO-SCU
    contexts = VerificationPresentationContexts + self.supported_contexts
    self.ae.supported_contexts = contexts

    self.ae.require_called_aet = self.require_called_aet
    self.ae.require_calling_aet = self.require_calling_aet

    # Handler setup
    # class needs to be instantiated before handlers can be defined
    self._evt_handlers = [
      (evt.EVT_C_STORE,  self._handle_c_store),
      (evt.EVT_ACCEPTED, self._handle_association_accepted),
      (evt.EVT_RELEASED, self._handle_association_released)
    ]
    self.updated_patients: Dict[Optional[int], Set[str]] = {}

    self.post_init()

  # End def __init__
  """ Store dataset process

  Responsiblities:
    - handle_c_store_message - extracts information from event
    - control_c_store_function - main function responsible for calling correct functions
  """
  def _handle_c_store(self, event: evt.Event) -> int:
    c_store_container = self._assocation_container_factory.build_assocation_c_store(event)
    return self.consume_c_store_container(c_store_container)

  def consume_c_store_container(self, c_store_container: CStoreContainer) -> int:
    try:
      if not self.filter(c_store_container.dataset):
        self.logger.warning("Dataset discarded")
        return 0xB006 # Element discarded
    except Exception as exception:
      log_traceback(self.logger, exception, "User flter")
      return 0xA801

    if self.patient_identifier_tag in c_store_container.dataset:
      patientID = deepcopy(c_store_container.dataset[self.patient_identifier_tag].value)
      try:
        self.data_state.add_image(c_store_container.dataset)
        self.updated_patients[c_store_container.assocation_id].add(patientID)
      except InvalidDataset:
        self.logger.info(f"Received dataset is not accepted by any inputs")
        return 0xB006
    else:
      self.logger.debug(f"Node: Received dataset, doesn't have patient Identifier tag")
      return 0xB007

    return 0x0000


  def _handle_association_accepted(self, event: evt.Event):
    """This is main handler for how the pynetdicom.AE should hanlde an
    evt.EVT_ACCEPTED event. You should be careful in overwriting this method.

    It creates a dataclass with all ness

    If you require different functionality, consider first if it's possible to
    extend the handler functions consume
    """
    self.logger.debug(f"Association with {event.assoc.requestor.ae_title} - {event.assoc.requestor.address} Accepted")
    association_accept_container = self._assocation_container_factory.build_assocation_accepted(event)

    for association_type in association_accept_container.assocation_types:
      handler = self.acceptation_handlers.get(association_type)
      if handler is not None:
        handler(self, association_accept_container)

  def consume_association_accept_store_assocation(
      self, accepted_container: AcceptedContainer):
    """This function completes all the calculation nessesarry for
    """
    if accepted_container.assocation_ip is not None:
      self._assocations_responds_addresses[accepted_container.assocation_id] = Address(
        accepted_container.assocation_ip, self.default_response_port, accepted_container.assocation_ae)

    self.updated_patients[accepted_container.assocation_id] = set()

  def _handle_association_released(self, event: evt.Event):
    """This function is called whenever an association is released

    It's the controller function for processing data

    Args:
        event (evt.Event):
    """
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Released.")
    released_container = self._assocation_container_factory.build_assocation_released(event)

    for association_type in released_container.assocation_types:
      handler = self.release_handlers.get(association_type)
      if handler is not None:
        handler(self, released_container)


  def consume_association_release_store_association(
      self, released_container: ReleasedContainer) -> None:
    """This function is called when an association, which stored
    some datasets is released.

    Args:
      released_container (ReleasedContainer): Dataclass containing information
        about the released association

    """
    for patient_ID in self.updated_patients[released_container.assocation_id]:
      if self.data_state.validate_patient_ID(patient_ID):
        self.logger.debug(f"Sufficient data for patient {patient_ID}")
        # Sadly my python Foo is not strong enough make a pretty solution here
        if self.processing_directory is not None:
          with TemporaryWorkingDirectory(self.processing_directory / str(patient_ID)) as twd:
            self._pipeline_processing(patient_ID, released_container)
        else:
          self._pipeline_processing(patient_ID, released_container)
      else:
        self.logger.debug(f"Insufficient data for patient {patient_ID}")
    del self.updated_patients[released_container.assocation_id] # Removing updated Patients

  def _pipeline_processing(self, patient_ID: str, released_container: ReleasedContainer):
    """Processes a patient through the pipeline and starts exporting it

    Args:
      patient_ID (str): Indentifier of the patient to be procesed
      released_container: (ReleasedContainer): data proccessing starts
        after an assocation is released. This is the data from the released
        association.
    """
    try:
      patient_input_container = self.get_input_container(patient_ID, released_container)
      result = self.process(patient_input_container)
    except Exception as exception:
      log_traceback(self.logger, exception, "processing")
    else:
      self.logger.debug(f"Process {patient_ID} Successful, Dispatching output!")
      if self._dispatch(result):
        self.logger.debug("Dispatching Successful")
        self.data_state.remove_patient(patient_ID)
      else:
        self.logger.error("Unable to dispatch pipeline output")

  def _dispatch(self, output: PipelineOutput) -> bool:
    """This function is responsible for triggering exporting of data and handling errors.
      You should consider if it's possible to create your own output rather than overwriting this function

      Args:
        output: PipelineOutput
      Returns:
        bool - If the output was successful in exporting the data.
    """
    try:
      success = output.send()
    except Exception as exception:
      log_traceback(self.logger, exception, "Output send function")
      success = False
    return success

  def get_input_container(self, patient_ID: str, released_container: ReleasedContainer) -> InputContainer:
    """This function retrives an input container for processing and
    fills out any information unavailable at object creation.

    Args:
      patient_ID (str): ID of the patient who data is in the input container
      released_container (ReleasedContainer): dataclass with relevant
        information from when event was released.
    """
    input_container = self.data_state.get_patient_input_container(patient_ID)

    if released_container.assocation_ae_title in self.known_endpoints:
      input_container.responding_address = self.known_endpoints[released_container.assocation_ae_title]
    elif released_container.assocation_id in self._assocations_responds_addresses:
      input_container.responding_address = self._assocations_responds_addresses[released_container.assocation_id]

    return input_container


  ##### User functions ######
  # These are the fucntions you should consider overwritting
  def filter(self, dataset : Dataset) -> bool:
    """This is a custom filter function, it is called before the node attempt to add the picture.

    Args:
        dataset pydicom.Dataset: Dataset, that this function determines the validity of.

    Returns:
        bool: if the dataset is valid, if True it'll attempt to add it, if not it'll send a 0xB006 response.
    """
    return True

  def process(self, input_data: InputContainer) -> PipelineOutput:
    """Function responsible for doing post processing.

    Args:
      input_data: InputContainer - Acts much like a dict similar to the input attribute
        containing the return data of the various input grinder functions
    Returns:
      PipelineOutput - The processed images of the pipeline
    """

    return NoOutput() # pragma: no cover

  def post_init(self) -> None:
    """This function is called just before the server is started.
      The idea being that a user change this function to run some arbitrary code before the Dicom node starts.
      This would often be

    Args:
        start (bool): Indication if the server should start
    """
    pass


  ##### Opening and closing. If you're overwritting these function, you should call super!
  def close(self) -> None:
    """Closes all connections active connections & cleans up any temporary working spaces.

      If your application includes additional connections, you should overwrite this method,
      And close any connections and call the super function.
    """
    self.logger.info("Closing Server!")
    if self.processing_directory is not None:
      chdir(self.__cwd)
      shutil.rmtree(self.processing_directory)

    self._maintenance_thread.stop()

    self.ae.shutdown()


  def open(self, blocking=True) -> Optional[NoReturn]:
    """Opens all connections active connections.
      If your application includes additional connections, you should overwrite this method,
      And open any connections and call the super function.

      Keyword Args:
        blocking (bool) : if true, this functions doesn't return.
    """
    if self.processing_directory is not None:
      self.__cwd = getcwd()
      if not self.processing_directory.exists():
        # Multiple Threads might attempt to create the directory at the same time
        self.processing_directory.mkdir(exist_ok=True)
      chdir(self.processing_directory)

    self._maintenance_thread.start()
    self.logger.info(f"Starting Server at port: {self.port} and AE: {self.ae_title}")
    self.ae.start_server(
      (self.ip,self.port),
      block=blocking,
      evt_handlers=self._evt_handlers)


  ##### Handler Directories #####
  # Handlers are an extendable way of

  # Extendable Handlers
  acceptation_handlers = { # Dict[AssociationTypes, Callable[[Self, AcceptedContainer], None]] # Note that Self type is only a part of python 3.11
    AssociationTypes.StoreAssocation : consume_association_accept_store_assocation
  }

  release_handlers = { # Dict[AssociationTypes, Callable[[Self, ReleasedContainer], None]]
    AssociationTypes.StoreAssocation : consume_association_release_store_association
  }


class AbstractQueuedPipeline(AbstractPipeline):
  """A pipeline that processes each object one at a time

  This might be very relevant when processing require a resource, such as GPU
  """
  process_queue: Queue[ReleasedContainer]

  queue_timeout = 0.05

  def process_worker(self):
    """Worker function for the process_queue"""
    while self.running:
      try:
        released_container = self.process_queue.get(timeout=self.queue_timeout)
        try:
          for association_type in released_container.assocation_types:
            handler = self.release_handlers.get(association_type)
            if handler is not None:
              handler(self, released_container)
        except Exception as exception:
          pass
        finally:
          self.logger.info("Finished queued task")
          self.process_queue.task_done()
      except Empty as E:
        pass

  def _handle_association_released(self, event: evt.Event):
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Released.")
    released_container = self._assocation_container_factory.build_assocation_released(event)

    self.process_queue.put(released_container)


  def __init__(self) -> None:
    self.running = True

    self.process_queue = Queue()
    self.process_thread = Thread(target=self.process_worker, daemon=False)
    self.process_thread.start()
    # Super is called at the end of the function as it might not return
    super().__init__()

  def close(self) -> None:
    self.running = False
    self.process_queue.join()

    return super().close()



class AbstractThreadedPipeline(AbstractPipeline):
  """Pipeline that creates threads to handle storing, to minimize IO load
  """
  threads: Dict[Optional[int],List[Thread]] = {}

  def _handle_c_store(self, event: evt.Event) -> int:
    thread: Thread = Thread(target=super()._handle_c_store, args=[event], daemon=False)
    thread.start()
    if event.assoc.native_id in self.threads:
      self.threads[event.assoc.native_id].append(thread)
    else:
      self.threads[event.assoc.native_id] = [thread]
    return 0x0000

  def join_threads(self, assoc_name:Optional[int] = None) -> None:
    if assoc_name is None:
      for thread_list in self.threads.values():
        for thread in thread_list: # pragma: no cover
          thread.join() # pragma: no cover
      self.threads = {}
    else:
      thread_list = self.threads[assoc_name]
      for thread in thread_list:
        thread.join()
      del self.threads[assoc_name]

  def _handle_association_released(self, event: evt.Event):
    self.join_threads(event.assoc.native_id)
    return super()._handle_association_released(event)
