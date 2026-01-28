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
from datetime import datetime, timedelta
from enum import Enum
import logging
from logging import getLogger, LogRecord
import multiprocessing
from multiprocessing import Queue as MPQueue
import os
from os import chdir, getcwd, getpid
from pathlib import Path
from queue import Queue, Empty
import shutil
import signal
from sys import stdout
from threading import Thread, Lock
from time import sleep, time
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, TextIO,\
  Type, Union, Tuple
from psutil import Process as PS_UTIL_Process

# Third part packages
from pydicom import Dataset
from pynetdicom import evt
from pynetdicom.ae import ApplicationEntity
from pynetdicom.presentation import AllStoragePresentationContexts, PresentationContext,\
  VerificationPresentationContexts
from pynetdicom.transport import ThreadedAssociationServer

# Dicomnode packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER
from dicomnode.dicom.dicom_factory import Blueprint, DicomFactory
from dicomnode.dicom.dimse import Address, send_image, DIMSE_StatusCodes
from dicomnode.dicom.series import DicomSeries
from dicomnode.lib.utils import spawn_process
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured,\
  CouldNotCompleteDIMSEMessage
from dicomnode.lib.io import Directory, File
from dicomnode.lib.logging import queue_logger_thread_target, set_logger,\
  LoggerConfig, log_traceback
from dicomnode.server.factories.association_events import AcceptedEvent, \
  AssociationEventFactory, AssociationTypes, CStoreEvent, ReleasedEvent
from dicomnode.server.input import AbstractInput
from dicomnode.server.pipeline_tree import PipelineTree, InputContainer, PatientNode
from dicomnode.server.maintenance import MaintenanceThread
from dicomnode.server.output import PipelineOutput, NoOutput
from dicomnode.server.processor import AbstractProcessor, ProcessRunnerArgs

default_sigint_handler = signal.getsignal(signal.SIGINT)

multiprocessing_context = multiprocessing.get_context('spawn')

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

  run_file: Optional[Union[str, Path]] = None

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

  Processor: Type[AbstractProcessor] = AbstractProcessor

  # End of Attributes definitions.
  def exporting_logging_config(self):
    if isinstance(self.log_output,str):
      output = Path(self.log_output)
    else:
      output = self.log_output

    return LoggerConfig(
      log_level=self.log_level,
      date_format=self.log_date_format,
      format=self.log_format,
      log_output=output,
      when=self.log_when,
      number_of_backups=self.number_of_backups,
    )

  def queue_logging_config(self):
    return LoggerConfig(
      log_level=self.log_level,
      date_format=self.log_date_format,
      format=self.log_format,
      log_output=self._log_queue,
      when=self.log_when,
      number_of_backups=self.number_of_backups
    )


  def _setup_logger(self):
    self.logger = getLogger(DICOMNODE_LOGGER_NAME)

    # If the logger is empty fill it with the current config
    if not len(self.logger.handlers):
      self._owns_dicomnode_logger = True
      set_logger(self.logger, self.exporting_logging_config())
    else:
      self._owns_dicomnode_logger = False

    self.__process_logger = getLogger(DICOMNODE_PROCESS_LOGGER)

    # Set pynetdicom logger
    getLogger("pynetdicom").setLevel(self.pynetdicom_logger_level)

  def _start_queue_logging(self):
    set_logger(self.logger, self.queue_logging_config())
    if self._owning_queue:
      set_logger(self.__process_logger, self.exporting_logging_config())

      self._logger_thread = Thread(
        target=queue_logger_thread_target,
        args=(self._log_queue,self.__process_logger),
        daemon=True
      )
      self._logger_thread.start()

  def _stop_queue_logging(self):
    set_logger(self.logger, self.exporting_logging_config())
    self.__process_logger.handlers.clear()
    if self._owning_queue:
      self._log_queue.put_nowait(None)
      self._logger_thread.join()
      self._log_queue.join_thread()


  def __init__(self, config=None) -> None:
    # This function starts and opens the server
    #
    # 1. Logging
    # 2. File system
    # 3. Create Server

    # logging
    if not hasattr(self, '_log_queue'):
      self._owning_queue = True
      self._log_queue: MPQueue[LogRecord | None] = multiprocessing_context.Queue()
    else:
      self._owning_queue = False
    self._setup_logger()

    self.__cwd = getcwd()
    # Load any previous state
    self._data_directory = Directory(self.data_directory) if self.data_directory is not None else None
    self._processing_directory = Directory(self.processing_directory) if self.processing_directory is not None else None

    if self._data_directory is not None and self._data_directory == self._processing_directory:
      raise IncorrectlyConfigured("data directory and processing directory cannot be equal")

    pipeline_tree_options = self.pipeline_tree_type.Options(
      ae_title=self.ae_title,
      data_directory=self._data_directory,
      lazy=self.lazy_storage,
      input_container_type=self.input_container_type,
      patient_container=self.patient_container_type,
    )

    self.data_state: PipelineTree = self.pipeline_tree_type(
      self.patient_identifier_tag,
      self.input,
      pipeline_tree_options
    )

    self.is_open = False

    self._association_event_factory = self.association_container_factory()

    # Server validations and creation.
    self.dicom_application_entry = ApplicationEntity(ae_title = self.ae_title)
    self.server_thread: ThreadedAssociationServer | None = None
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

    if self.Processor is AbstractProcessor or not issubclass(self.Processor, AbstractProcessor):
      raise IncorrectlyConfigured("Missing Processor class!")

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

  def _handle_c_echo(self, event: evt.Event):
    self.logger.debug(f"Connection {event.assoc.remote['ae_title']} send an echo") #type: ignore
    return 0x0000

  def _handle_connection_opened(self, event: evt.Event):
    self.logger.debug(f"Connection {event.address[0]} opened a connection") #type: ignore

  def _handle_connection_closed(self, event: evt.Event):
    self.logger.debug(f"Connection {event.address[0]} closed a connection") #type: ignore
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Closed a connection.")

    released_event = self._association_event_factory.build_association_released(event)

    for association_type in released_event.association_types:
      handler = self._release_handlers.get(association_type, None)
      if handler is not None:
        handler(self, released_event)
      else:
        self.logger.critical(f"Missing Release Handler for {association_type}!") # pragma: no cover

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

    return


  def _release_store_handler(self, released_event: ReleasedEvent):
    input_containers = self._extract_input_containers(released_event)
    if len(input_containers) == 0:
      self.logger.info(f"Release Event from {released_event.association_ae} did"
                       " not return any input containers to be processed.")
      return

    for patient_id, input_container in input_containers:
      args = ProcessRunnerArgs(
        patient_id=patient_id,
        input_container=input_container,
        log_config=self.queue_logging_config(),
        process_path=self.get_processing_directory_path(patient_id)
      )

      timeouts = 0
      started_process = datetime.now()
      timeout_seconds = 1.0


      if self.processing_directory is not None:
        process = spawn_process(
          self.Processor, args, context=multiprocessing_context
        )

        while process.is_alive():
          process.join(timeout_seconds)

          if process.is_alive():
            timeouts += 1
            self.logger.info(f"Process {process.pid} started at {started_process.time()} encountered timeout {timeouts}")
            self.logger.info(f"Process {process.pid} has status: {PS_UTIL_Process(process.pid).status}")
            timeout_seconds = (timeout_seconds * 2)

        if process.exitcode:
          self.logger.critical(f"Sub process failed with exitcode {process.exitcode}")
        else:
          patients_to_clean_up = [fst for fst, sec in input_containers]
          self.logger.info(f"Removing {patients_to_clean_up}'s images")
          self.data_state.clean_up_patients(patients_to_clean_up)
      else:
        thread = Thread(target=self.Processor, args=(args,), name="DicomnodeThreadName") # There might no reason to spawn a thread here
        thread.start()
        while thread.is_alive():
          thread.join(timeout=timeout_seconds)
          if thread.is_alive():
            timeouts += 1
            self.logger.info(f"Thread {thread.native_id} started at {started_process.time()} encountered timeout {timeouts}")
            timeout_seconds = (timeout_seconds * 2)

          patients_to_clean_up = [fst for fst, sec in input_containers]
          self.logger.info(f"Removing {patients_to_clean_up}'s images")
          self.data_state.clean_up_patients(patients_to_clean_up)


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
    if not self.is_open:
      self.logger.error("Attempted to close an closed node")
      return

    # Gracefully shutdown is fucking hard...
    while self.dicom_application_entry.active_associations != []: #pragma: no cover
      sleep(0.005)

    self.dicom_application_entry.shutdown()
    if self.server_thread is not None:
      self.server_thread.server_close()

    self.logger.info("Closing Server!")
    if self.processing_directory is not None:
      chdir(self.__cwd)
      shutil.rmtree(self.processing_directory)

    self._maintenance_thread.stop()
    self._stop_queue_logging()


  def open(self, blocking=True) -> Optional[NoReturn]:
    """Opens all connections active connections.
      If your application includes additional connections, you should overwrite this method,
      And open any connections and call the super function.

      Keyword Args:
        blocking (bool) : if true, this functions doesn't return.
    """
    if self.is_open:
      self.logger.error("Attempted to open a node, that was already open")
      return

    self.logger.info(f"Starting Server at address: {self.ip}:{self.port} and AE: {self.ae_title}")
    self.logger.debug(f"self.dicom_application_entry.require_called_aet: {self.dicom_application_entry.require_called_aet}")
    self.logger.debug(f"self.dicom_application_entry.require_calling_aet: {self.dicom_application_entry.require_calling_aet}")
    self.logger.debug(f"self.dicom_application_entry.maximum_pdu_size: {self.dicom_application_entry.maximum_pdu_size}")
    self.logger.debug(f"The loaded initial pipeline tree is: {self.data_state}")

    signal.signal(signal.SIGINT, self.node_signal_handler_SIGINT)
    self._start_queue_logging()

    if self.processing_directory is not None:
      if not self.processing_directory.exists():
        self.processing_directory.mkdir(exist_ok=True)
      chdir(self.processing_directory)

    self._maintenance_thread = self.maintenance_thread(
      self.data_state, self.study_expiration_days, daemon=True
    )
    self._maintenance_thread.start()

    #self.logger.debug(f"self.dicom_application_entry.supported_contexts: {self.dicom_application_entry.supported_contexts}")
    # I don't know why the type checker is high.
    event_handlers: List[evt.EventHandlerType] = [ t for t in self._evt_handlers.items() ]

    self.is_open = True
    self.server_thread = self.dicom_application_entry.start_server(
      (self.ip,self.port),
      block=blocking,
      evt_handlers=event_handlers
    )

  class ConnectionContextManager():
    def __init__(self, node: 'AbstractPipeline'):
      self.node = node

    def __enter__(self):
      self.node.open(blocking=False)

    def __exit__(self, exception_type, exception_value, exception_traceback):
      self.node.close()

  def open_cm(self):
    return self.ConnectionContextManager(self)

  def get_processing_directory_path(self, identifier) -> Optional[Path]:

    if self._processing_directory is None:
      return None

    if isinstance(identifier, DicomSeries):
      val = identifier[self.patient_identifier_tag]
      if val is not None and not isinstance(val, List):
        return self._processing_directory / str(val.value)
      else:
        return None

    if isinstance(identifier, Dataset):
      return self._processing_directory / str(identifier[self.patient_identifier_tag].value)

    if isinstance(identifier, str):
      return self._processing_directory / identifier

    raise TypeError("Unknown identifier type")

  def get_storage_directory(self, identifier: Any) -> Optional[Directory]:
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

  def node_signal_handler_SIGINT(self, signal_, frame): #pragma: no cover
    """Signal handler for a running dicom node. Kills alls children processes.

    Args:
        signal_ (signal.Signame): The signal that this process should send to it
          children
        frame (_type_): I have no clue what this arg is
    """
    # this code is tested, but because it's not run covering
    pid = getpid()

    self.logger.critical(f"Process {pid} Received Signal {signal_}")
    attempts = 0
    while children := PS_UTIL_Process().children(recursive=True):
      if 5 < attempts:
        break
      for child in children:
        self.logger.critical(f"Send signal {signal_} to {child.pid} - {child.status()}")
        if child.is_running():
          child.send_signal(signal_)
      attempts += 1
      sleep(0.015)
    for child in PS_UTIL_Process().children(recursive=True):
      # If sending signals doesn't work, NUKING TIME!
      self.logger.critical(f"BRUTALLY KILLED {child.pid}")
      child.kill()

    self.logger.critical(f"Killed all subprocesses!")

    if signal_ in [signal.SIGTERM, signal.SIGINT] and self._owning_queue:
      self._log_queue.put(None, timeout=1.0)

      self._logger_thread.join()
    exit(1)


  def process_signal_handler_SIGINT(self, signal_, frame): #pragma: no cover
    # Same as above
    self.logger.critical("Process killed by Parent")
    exit(1)


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

  def node_signal_handler_SIGINT(self, signal_, frame):
    self.running = False
    super().node_signal_handler_SIGINT(signal_, frame)

  def __init__(self) -> None:
    self.running = True

    # Super is called at the end of the function
    super().__init__()
    self._evt_handlers[evt.EVT_RELEASED] = self._handle_association_released

  def open(self, blocking=True):
    signal.signal(signal.SIGINT, self.node_signal_handler_SIGINT)

    self.process_queue = Queue()
    self.process_thread = Thread(target=self.process_worker, daemon=False)
    self.process_thread.start()
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
