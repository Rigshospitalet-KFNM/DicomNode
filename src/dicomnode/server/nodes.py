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
from enum import Enum
import logging
from logging import getLogger
import multiprocessing
from os import chdir, getcwd, kill
from pathlib import Path
from queue import Queue, Empty
import shutil
import signal
from sys import stdout
from threading import Thread, Lock
from time import sleep
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, TextIO,\
  Type, Union, Tuple
from psutil import Process as PS_UTIL_Process

# Third part packages
from pynetdicom import evt
from pynetdicom.ae import ApplicationEntity
from pynetdicom.presentation import AllStoragePresentationContexts, PresentationContext,\
  VerificationPresentationContexts
from pydicom import Dataset

# Dicomnode packages
from dicomnode.dicom.dicom_factory import Blueprint, DicomFactory
from dicomnode.dicom.dimse import Address, send_image, DIMSE_StatusCodes
from dicomnode.dicom.series import DicomSeries
from dicomnode.lib import config_parser
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured,\
  CouldNotCompleteDIMSEMessage
from dicomnode.lib.io import TemporaryWorkingDirectory
from dicomnode.lib.logging import log_traceback, set_logger, get_logger
from dicomnode.server.factories.association_events import AcceptedEvent, \
  AssociationEventFactory, AssociationTypes, CStoreEvent, ReleasedEvent
from dicomnode.server.input import AbstractInput
from dicomnode.server.pipeline_tree import PipelineTree, InputContainer, PatientNode
from dicomnode.server.maintenance import MaintenanceThread
from dicomnode.server.output import PipelineOutput, NoOutput

default_sigint_handler = signal.getsignal(signal.SIGINT)

class ProcessingDirectoryOptions(Enum):
  NO_PROCESSING_DIRECTORY = 0
  USE_DATA_DIRECTORY = 1


class AbstractPipeline():
  """Base Class for an image processing Pipeline
  Creates a SCP server and starts it, unless start=False is passed

  Should be subclassed with your implementation of the process function at least.

  Check tutorials/ConfigurationOverview.md for an overview of attributes
  """

  # Directory for file Processing
  processing_directory: Optional[Path] = None
  """Base directory that the processing will take place in.
  The specific directory that will run is: `processing_directory/patient_ID`

  DO NOT SET THIS EQUAL TO DATA DIRECTORY
  """

  # Maintenance Configuration
  maintenance_thread: Type[MaintenanceThread] = MaintenanceThread
  """Class of MaintenanceThread to be created when the server opens"""

  study_expiration_days: int = 14
  """The amount of days a study will hang in memory,
  before being clean up by the MaintenanceThread"""


  # Input configuration
  input: Dict[str, Type[AbstractInput]] = {}
  "Defines the AbstractInput leafs for each Patient Node."

  patient_identifier_tag: int = 0x00100020 # Patient ID
  "Dicom tag to separate each study in the PipelineTree"

  data_directory: Optional[Path] = None
  """If None the pipeline will only store data in memory before processing.
  If Path, then the pipeline store root of the tree at that path. Creates folder
  if path doesn't exists.

  Required to be a path if lazy_storage=True
  """

  lazy_storage: bool = False
  "Indicates if the abstract inputs should use Lazy datasets or not"

  pipeline_tree_type: Type[PipelineTree] = PipelineTree
  "Class of PipelineTree that the node will create as main data storage"

  patient_container_type: Type[PatientNode] = PatientNode
  "Class of PatientNode that the the PipelineTree should create as nodes."

  input_container_type: Type[InputContainer] = InputContainer
  "Class of PatientContainer that the PatientNode should create when processing a patient"

  # Dicom communication configuration tags
  ae_title: str = "Your_AE_TITLE"
  "AE title of the dicomnode"

  ip: str = "localhost"
  "IP of node either 0.0.0.0 or localhost"

  port: int = 104
  "Port of Node, int in range 1-65535 (Requires root access to open port <1024)"

  supported_contexts: List[PresentationContext] = AllStoragePresentationContexts
  "Presentation contexts accepted by the node"

  require_called_aet: bool = True
  "Require caller to specify AE title of node"

  require_calling_aet: List[str] = []
  "If not empty require the node only to accept connection from AE titles in this attribute"

  known_endpoints: Dict[str, Address] = {}
  "Address book indexed by AE titles."

  _associations_responds_addresses: Dict[int, Address]
  "Internal variable containing a mapping of association to endpoint address"

  association_container_factory: Type[AssociationEventFactory] = AssociationEventFactory
  """Class of Factory, that extracts information from the association to the underlying
  processing function."""

  default_response_port: int = 104
  "Default Port used for unspecified Dicomnodes"

  unhandled_error_blueprint: Optional[Blueprint] = None

  #Logging Configuration
  number_of_backups: int = 8
  "Number of backups before the os starts deleting old logs"

  log_date_format: str = "%Y/%m/%d %H:%M:%S"
  "String format for timestamps in logs."

  log_output: Optional[Union[TextIO, Path, str]] = stdout
  """Destination of log output:
  * `None` - Disables The logger
  * `TextIO` - output to that stream, This is stdout / stderr
  * `Path | str` - creates a rotating log at the path
  """

  log_when: str = "w0"
  "At what points in time the log should roll over, defaults to monday midnight"

  log_level: int = logging.INFO
  "Level of Logger"

  log_format: str = "[%(asctime)s] |%(thread_id)d| %(name)s - %(levelname)s - %(message)s"
  "Format of log messages using the '%' style."

  pynetdicom_logger_level: int = logging.CRITICAL + 1
  """Sets the level pynetdicom logger, note that traceback from
  associations are logged to pynetdicom, which can be helpful for bugfixing"""

  error_on_rejected_dataset = True
  """If true the server will send a error message code back to the server if no
  dataset were accepted by the inputs, if false it will send an accepted image
  """

  # End of Attributes definitions.
  def _setup_logger(self):
    if isinstance(self.log_output, str):
      self.log_output = Path(self.log_output)

    self.logger = set_logger(
      log_output=self.log_output,
      log_level=self.log_level,
      format=self.log_format,
      date_format=self.log_date_format,
      backupCount=self.number_of_backups,
      when=self.log_when
    )
    # Set pynetdicom logger
    getLogger("pynetdicom").setLevel(self.pynetdicom_logger_level)


  def __init__(self, config=None) -> None:
    # This function starts and opens the server
    #
    # 1. Logging
    # 2. File system
    # 3. Create Server

    # logging
    self._setup_logger()

    self.__cwd = getcwd()
    # Load any previous state
    if self.data_directory is not None:
      if not isinstance(self.data_directory, Path):
        self.data_directory = Path(self.data_directory)

      if self.data_directory.is_file():
        raise IncorrectlyConfigured("The root data directory exists as a file.")

      if not self.data_directory.exists():
        self.data_directory.mkdir(parents=True)

      if self.data_directory == self.processing_directory:
        raise IncorrectlyConfigured("data directory and processing directory cannot be equal")

    pipeline_tree_options = self.pipeline_tree_type.Options(
      ae_title=self.ae_title,
      data_directory=self.data_directory,
      lazy=self.lazy_storage,
      input_container_type=self.input_container_type,
      patient_container=self.patient_container_type,
    )

    self.data_state: PipelineTree = self.pipeline_tree_type(
      self.patient_identifier_tag,
      self.input,
      pipeline_tree_options
    )

    self._maintenance_thread = self.maintenance_thread(
      self.data_state, self.study_expiration_days, daemon=True)

    self._association_event_factory = self.association_container_factory()

    # Server validations and creation.
    self.dicom_application_entry = ApplicationEntity(ae_title = self.ae_title)
    # You need VerificationPresentationContexts for ECHOSCU
    # and you want ECHO-SCU
    contexts = VerificationPresentationContexts + self.supported_contexts
    self.dicom_application_entry.supported_contexts = contexts

    self.dicom_application_entry.require_called_aet = self.require_called_aet
    self.dicom_application_entry.require_calling_aet = self.require_calling_aet

    # Handler setup
    # class needs to be instantiated before handlers can be defined
    self._associations_responds_addresses = {}
    self._updated_patients: Dict[Optional[int], Dict[str, int]] = {}
    self._patient_locks: Dict[str, Tuple[Set[int], Lock]] = {}
    self._lock_key = Lock()

    # It's import that this is initialized here, because otherwise the self
    # argument is not passed properly.
    # If you need to replace these handler it's important to call super's init
    # then overwrite the handler function
    self._evt_handlers: Dict[evt.EventType, Callable ] = {
      evt.EVT_C_ECHO     : self._handle_c_echo,
      evt.EVT_CONN_OPEN  : self._handle_connection_opened,
      evt.EVT_CONN_CLOSE : self._handle_connection_closed,
      evt.EVT_C_STORE    : self._handle_c_store,
      evt.EVT_ACCEPTED   : self._handle_association_accepted,
      evt.EVT_REJECTED   : self._handle_association_rejected,
      evt.EVT_RELEASED   : self._handle_association_released,
    }

    self.post_init()
  # End def __init__
  #region logging handlers


  def _handle_c_echo(self, event: evt.Event):
    self.logger.debug(f"Connection {event.assoc.remote['ae_title']} send an echo") #type: ignore
    return 0x0000

  def _handle_connection_opened(self, event: evt.Event):
    self.logger.debug(f"Connection {event.address[0]} opened a connection") #type: ignore

  def _handle_connection_closed(self, event: evt.Event):
    self.logger.debug(f"Connection {event.address[0]} closed a connection") #type: ignore

  def _handle_association_rejected(self, event: evt.Event):
    self.logger.debug(f"Connection {event.assoc.remote['ae_title']} rejected a connection") #type: ignore

  # Store dataset process
  # Responsibility's:
  #  - handle_c_store_message - extracts information from event
  #  - control_c_store_function - main function responsible for calling correct functions
  def _handle_association_accepted(self, event: evt.Event):
    """This is main handler for how the pynetdicom.AE should hanlde an
    evt.EVT_ACCEPTED event. You should be careful in overwriting this method.

    It creates a dataclass with all ness

    If you require different functionality, consider first if it's possible to
    extend the handler functions consume
    """
    self.logger.debug(f"Association with {event.assoc.requestor.ae_title}"
                      f" - {event.assoc.requestor.address} Accepted")
    association_accept_container = self._association_event_factory\
                                       .build_association_accepted(event)

    for association_type in association_accept_container.association_types:
      handler = self._acceptation_handlers.get(association_type)
      if handler is not None:
        handler(self, association_accept_container)
      else:
        self.logger.error("No association requested handler for found for "
                          f"{association_type}!") # pragma no cover


  def _consume_association_accept_store_association(
      self, accepted_container: AcceptedEvent):
    """This function initialized after a thread have connected
    """
    if accepted_container.association_ip is not None:
      self._associations_responds_addresses[accepted_container.association_id] = Address(
        accepted_container.association_ip,
        self.default_response_port,
        accepted_container.association_ae
      )

    self._updated_patients[accepted_container.association_id] = {}


  def _handle_c_store(self, event: evt.Event) -> int:
    c_store_container = self._association_event_factory.build_association_c_store(event)
    status = self.consume_c_store_container(c_store_container)
    return status

  def consume_c_store_container(self, c_store_container: CStoreEvent) -> int:
    try:
      if not self.filter(c_store_container.dataset):
        self.logger.info("Node rejected dataset: Received Dataset did not pass filter")
        return 0xB006 # Element discarded
    except Exception as exception:
      log_traceback(self.logger, exception, "User defined function filter produced an exception")
      return 0xA801

    if self.patient_identifier_tag in c_store_container.dataset:
      patient_id = str(c_store_container.dataset[self.patient_identifier_tag].value)
      thread_id = c_store_container.association_id
      # Critical zone for managing keys
      with self._lock_key:
        if patient_id in self._patient_locks:
          threads, patient_lock = self._patient_locks[patient_id]
        else:
          threads, patient_lock = (set([thread_id]), Lock())
          self._patient_locks[patient_id] = (threads, patient_lock)

        if patient_id not in self._updated_patients[thread_id]:
          self._updated_patients[thread_id][patient_id] = 0
          threads.add(thread_id)
      # End of Critical zone
      try:
        # Critical Patient zone
        with patient_lock:
          stored = self.data_state.add_image(c_store_container.dataset)
          self._updated_patients[thread_id][patient_id] += stored
        # End of Critical Zone
      except InvalidDataset:
        self.logger.info("Node rejected dataset: Received dataset is not accepted by any inputs")
        self.logger.info(f"SOP class: {c_store_container.dataset[0x0008_0016]}")
        if self.error_on_rejected_dataset:
          return 0xB006
        else:
          return 0x0000

      except Exception as exception:
        log_traceback(self.logger, exception, "Adding Image to input produced an exception")
        return 0xA801
    else:
      self.logger.info(f"Node rejected dataset: Received dataset doesn't have patient Identifier tag: {hex(self.patient_identifier_tag)}")
      return 0xB007

    return 0x0000

  def _extract_input_containers(self, release_event: ReleasedEvent
                                ) -> List[Tuple[str, InputContainer]]:
    """Iterates through all patients of the pipeline tree that this association
    have added images to and extract all valid inputs from pipeline tree of this
    node.

    Side Effects:
      Removes the association id (thread id) from self._updated_patients

    Args:
        release_event (ReleasedEvent): The Event triggered by an C-STORE
        association releasing (finishing storing images in the dicom node)

    Returns:
        List[Tuple[str, InputContainer]]: A list of all patients that is
    """
    input_containers: List[Tuple[str, InputContainer]] = []
    self.logger.debug(f"PatientID to be updated in: {self._updated_patients}")
    with self._lock_key:
      for patient_id in self._updated_patients[release_event.association_id]:
        if patient_id in self._patient_locks:
          threads, patient_lock = self._patient_locks[patient_id]
        else:
          self.logger.critical("This is a bug in the library, please report it") #pragma: no cover
          self.logger.critical(f"patient_locks: {self._patient_locks} patient id: {patient_id}") #pragma: no cover
          self.logger.critical("Another thread deleted thread-set and Patient log") #pragma: no cover
          continue # pragma: no cover
        self.logger.info(f"Thread {release_event.association_id} added: {self._updated_patients[release_event.association_id][patient_id]} images")
        with patient_lock:
          if len(threads) == 1:
            if self.data_state.validate_patient_id(patient_id):
              # Note this prevents you from adding more images to that patient
              # While the other locks prevents multiple threads from adding
              input_containers.append(
                (patient_id, self._get_input_container(patient_id, release_event))
              )
              del self._patient_locks[patient_id]
            else:
              self.logger.info(f"Insufficient data for patient {patient_id}")
              del self._patient_locks[patient_id]
              continue
          else:
            thread_id = release_event.association_id
            threads.remove(thread_id)
            self.logger.debug(f"One of the Threads: {threads} will handle {patient_id}")
            continue
    del self._updated_patients[release_event.association_id] # Removing updated Patients
    return input_containers


  #region handle_association_released
  def _handle_association_released(self, event: evt.Event):
    """This function is called whenever an association is released

    It's the controller function for processing data

    Args:
        event (evt.Event):
    """
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Released.")
    released_event = self._association_event_factory.build_association_released(event)

    for association_type in released_event.association_types:
      handler = self._release_handlers.get(association_type, None)
      if handler is not None:
        handler(self, released_event)
      else:
        self.logger.critical(f"Missing Release Handler for {association_type}!") # pragma: no cover

  def _release_store_handler(self, released_event: ReleasedEvent):
    input_containers = self._extract_input_containers(released_event)
    if len(input_containers) == 0:
      self.logger.info(f"Release Event from {released_event.association_ae} did"
                       " not return any input containers to be processed.")
      return
    process = multiprocessing.Process(target=self._process_entry_point,
                                      args=(released_event, input_containers))
    process.start()
    process.join()
    if process.exitcode:
      self.logger.critical(f"Sub process failed with exitcode {process.exitcode}")
    else:
      patients_to_clean_up = [fst for fst, sec in input_containers]
      self.logger.info(f"Removing {patients_to_clean_up}'s images")
      self.data_state.clean_up_patients(patients_to_clean_up)

  def _process_entry_point(
      self, released_container, input_containers: List[Tuple[str, InputContainer]] ) -> None:
    """This function is called when an association, which stored
    some datasets is released.

    Args:
      released_container (ReleasedContainer): Dataclass containing information
        about the released association

    """
    self.logger = getLogger("dicomnode") # Reset loggers as
    for patient_id, input_container in input_containers:
      self.logger.debug(f"Started to process {patient_id}")
      processing_directory = self.get_processing_directory(patient_id)
      if processing_directory is not None:
        with TemporaryWorkingDirectory(processing_directory):
          self._pipeline_processing(patient_id, released_container, input_container)
      else:
        self._pipeline_processing(patient_id, released_container, input_container)


  def _pipeline_processing(self,
                           patient_id: str,
                           released_container: Optional[ReleasedEvent],
                           patient_input_container: InputContainer):
    """Processes a patient through the pipeline and exporting it

    This function is not called directly, it's a target process is spawned,
    when the data is spawned.

    Args:
      patient_ID (str): Identifier of the patient to be processed
      released_container: (ReleasedContainer): data processing starts
        after an association is released. This is the data from the released
        association.
    """
    self.logger.info(f"Processing {patient_id}")
    try:
      result = self.process(patient_input_container)
      if result is None:
        error_message = "You forgot to return a PipelineOutput object in the "\
                        "process function. If output is handled by process, "\
                        "return a NoOutput Object"
        self.logger.warning(error_message)
        result = NoOutput()
    except Exception as exception:
      log_traceback(self.logger, exception, "Exception in user Processing")
      exception_handler = self.exception_handlers.get(
        type(exception), self.exception_handler_respond_with_dataset
      )
      exception_handler(exception, released_container, patient_input_container)
      exit(1) # This is okay because this function is called from another process
    else:
      self.logger.debug(f"Process {patient_id} Successful, Dispatching output!")
      if self._dispatch(result):
        self.logger.info(f"Dispatched {patient_id} Successful")
      else:
        self.logger.error(f"Unable to dispatch output for {patient_id}")

  def _dispatch(self, output: PipelineOutput) -> bool:
    """This function is responsible for triggering exporting of data and handling errors.
      You should consider if it's possible to create your own output
      rather than overwriting this function

      Args:
        output: PipelineOutput - the output to be exported
      Returns:
        bool - If the output was successful in exporting the data.
    """
    try:
      success = output.send()
    except Exception as exception:
      log_traceback(self.logger, exception, "Exception in user Output Send Function")
      success = False
    return success

  def _get_input_container(self,
                           patient_id: str,
                           released_container: ReleasedEvent) -> InputContainer:
    """This function retrieves the patients input container for processing and
    fills out any information unavailable at object creation.

    Args:
      patient_ID (str): ID of the patient who data is in the input container
      released_container (ReleasedContainer): dataclass with relevant
        information from when event was released.
    """
    input_container = self.data_state.get_patient_input_container(patient_id)

    if released_container.association_ae in self.known_endpoints:
      input_container.responding_address = self.known_endpoints[
        released_container.association_ae
      ]
    elif released_container.association_id in self._associations_responds_addresses:
      input_container.responding_address = self._associations_responds_addresses[
        released_container.association_id]

    return input_container


  ##### User functions ######
  #region User functions
  # These are the functions you should consider overwriting
  def filter(self, dataset : Dataset) -> bool:
    """This is a custom filter function, it is called before the node attempt to add the picture.

    Args:
        dataset pydicom.Dataset: Dataset, that this function determines the validity of.

    Returns:
        bool: if the dataset is valid, if True it'll attempt to add it,
              if not it'll send a 0xB006 response.
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

    The idea being that a user change this function to run some arbitrary code
    before the Dicom node starts.

    Args:
        start (bool): Indication if the server should start
    """



  ##### Opening and closing. If you're overwriting these function, you should call super!
  def close(self) -> None:
    """Closes all connections active connections & cleans up any temporary working spaces.

      If your application includes additional connections, you should overwrite this method,
      And close any connections and call the super function.
    """
    while self.dicom_application_entry.active_associations != []: #pragma: no cover
      sleep(0.005)

    self.logger.info("Closing Server!")
    if self.processing_directory is not None:
      chdir(self.__cwd)
      shutil.rmtree(self.processing_directory)

    self._maintenance_thread.stop()

    self.dicom_application_entry.shutdown()


  def open(self, blocking=True) -> Optional[NoReturn]:
    """Opens all connections active connections.
      If your application includes additional connections, you should overwrite this method,
      And open any connections and call the super function.

      Keyword Args:
        blocking (bool) : if true, this functions doesn't return.
    """
    if self.processing_directory is not None:
      if not self.processing_directory.exists():
        # Multiple Threads might attempt to create the directory at the same time
        self.processing_directory.mkdir(exist_ok=True)
      chdir(self.processing_directory)

    self._maintenance_thread.start()
    self.logger.info(f"Starting Server at address: {self.ip}:{self.port} and AE: {self.ae_title}")
    self.logger.debug(f"self.dicom_application_entry.require_called_aet: {self.dicom_application_entry.require_called_aet}")
    self.logger.debug(f"self.dicom_application_entry.require_calling_aet: {self.dicom_application_entry.require_calling_aet}")
    self.logger.debug(f"self.dicom_application_entry.maximum_pdu_size: {self.dicom_application_entry.maximum_pdu_size}")

    self.logger.debug(f"The loaded initial pipeline tree is: {self.data_state}")
    #self.logger.debug(f"self.dicom_application_entry.supported_contexts: {self.dicom_application_entry.supported_contexts}")
    # I don't know why the type checker is high.
    event_handlers: List[evt.EventHandlerType] = [ t for t in self._evt_handlers.items() ]

    self.dicom_application_entry.start_server(
      (self.ip,self.port),
      block=blocking,
      evt_handlers=event_handlers)

  def get_processing_directory(self, identifier) -> Optional[Path]:
    if self.processing_directory is None:
      return None

    if isinstance(identifier, DicomSeries):
      val = identifier[self.patient_identifier_tag]
      if val is not None and not isinstance(val, List):
        return self.processing_directory / str(val.value)
      else:
        return None

    if isinstance(identifier, Dataset):
      return self.processing_directory / str(identifier[self.patient_identifier_tag].value)

    if isinstance(identifier, str):
      return self.processing_directory / identifier

    raise TypeError("Unknown identifier type")

  def get_storage_directory(self, identifier: Any) -> Optional[Path]:
    return self.data_state.get_storage_directory(identifier)

  def _build_error_dataset(self, blueprint: Blueprint,
                                 pivot: Dataset,
                                 exception: Exception) -> Dataset:
    factory = DicomFactory()
    return factory.build_instance(
      pivot, blueprint, kwargs={
        'exception' : exception
      }
    )


  def exception_handler_respond_with_dataset(self,
                                             exception : Exception,
                                             release_container: Optional[ReleasedEvent],
                                             input_container: InputContainer):
    """This function is the default exception handler for processing

    By default, if there's a error blueprint, it creates a dicom image, and
    sends it back to the triggering ip address.

    For more info read the tutorial on the error handling

    Args:
        exception (Exception): The unhandled Exception triggered from process
        release_container (ReleasedContainer): The Release Container pipeline
        processing was called with. Contains information on association
        input_container (InputContainer): The InputContainer that was used to
        call process that threw.
    """
    if self.unhandled_error_blueprint is None or release_container is None:
      return

    if release_container.association_ip is None:
      error_message = "Unable to send error dataset to client due to missing "\
                      "IP address"
      self.logger.error(error_message)
      return

    pivot_dataset = None
    for dataset_iterator in input_container.datasets.values():
      for dataset in dataset_iterator:
        pivot_dataset = dataset
        break
      if pivot_dataset is not None:
        break

    if pivot_dataset is None:
      self.logger.error("Unable to extract a dataset from the input container")
      self.logger.error("Unless this is a test case, I am impressed")
      return

    response_address = Address(release_container.association_ip,
                               self.default_response_port,
                               release_container.association_ae)

    error_dataset = self._build_error_dataset(
      self.unhandled_error_blueprint,
      pivot_dataset,
      exception
    )
    try:
      response = send_image(self.ae_title, response_address, error_dataset)
      if 0x0000_0900 in response: # Status tag
        if response.Status == DIMSE_StatusCodes.SUCCESS:
          self.logger.info(f"Client {response_address} is informed of Failure "
                           "triggered by Process Exception")
        else:
          self.logger.error(f"Client {response_address} did not accept the "
                            f"error dataset, and responded with {response}")
      else:
        self.logger.error(f"Client {response_address} send an invalid response...")
    except CouldNotCompleteDIMSEMessage:
      self.logger.error("Unable to send error message to the client at "
                        f"{response_address}")

  ##### Handler Directories #####
  # Handlers are an extendable way of

  # Extendable Handlers
  # Note that Self type is only a part of python 3.11
  _acceptation_handlers = { # Dict[AssociationTypes, Callable[[Self, AcceptedContainer], None]]
    AssociationTypes.STORE_ASSOCIATION : _consume_association_accept_store_association
  }

  _release_handlers = { # Dict[AssociationTypes, Callable[[Self, ReleasedContainer], None]]
    AssociationTypes.STORE_ASSOCIATION : _release_store_handler
  }

  exception_handlers: Dict[Type[Exception], Callable[[Exception, Optional[ReleasedEvent], InputContainer], None]] = {

  }

class AbstractQueuedPipeline(AbstractPipeline):
  """A pipeline that processes each object one at a time

  This might be very relevant when processing require a resource, such as GPU
  """
  process_queue: Queue[ReleasedEvent]

  queue_timeout = 0.05

  def process_worker(self):
    """Worker function for the process_queue thread"""
    while self.running:
      try:
        released_container = self.process_queue.get(timeout=self.queue_timeout)
        self.logger.info(f"Process queue got item, approximate queue length: "
                         f"{self.process_queue.qsize()}")
        try:
          for association_type in released_container.association_types:
            handler = self._release_handlers.get(association_type)
            if handler is not None:
              handler(self, released_container)
        except Exception as exception:
          log_traceback(self.logger, exception, "Process worker encountered exception")
        finally:
          self.process_queue.task_done()
      except Empty:
        pass

  def _handle_association_released(self, event: evt.Event):
    released_container = self._association_event_factory.build_association_released(event)
    self.process_queue.put(released_container)

  def signal_handler_SIGINT(self, signal, frame):
    self.running = False
    current_process = PS_UTIL_Process()
    for child in current_process.children(recursive=False):
      child.send_signal(signal)

  def __init__(self, master_queue: Optional[Queue[ReleasedEvent]]=None) -> None:
    self.running = True
    if master_queue is None:
      self.process_queue = Queue()
    else:
      self.process_queue = master_queue
    self.process_thread = Thread(target=self.process_worker, daemon=False)
    self.process_thread.start()

    # Super is called at the end of the function
    super().__init__()
    self._evt_handlers[evt.EVT_RELEASED] = self._handle_association_released

  def open(self, blocking=True):
    signal.signal(signal.SIGINT, self.signal_handler_SIGINT)
    super().open(blocking)

  def close(self) -> None:
    self.running = False
    self.process_queue.join()
    signal.signal(signal.SIGINT, default_sigint_handler)

    return super().close()

__all__ = (
  "AbstractPipeline",
  "AbstractQueuedPipeline",
)
