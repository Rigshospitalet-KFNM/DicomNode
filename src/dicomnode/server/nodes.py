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
from threading import Thread, Lock, get_native_id
from time import sleep
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
from dicomnode.dicom import DicomIdentifier, display_dicom_collection
from dicomnode.dicom.dicom_factory import Blueprint, DicomFactory
from dicomnode.dicom.dimse import Address, dataset_from_event
from dicomnode.dicom.series import DicomSeries
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured
from dicomnode.lib.parallelism import Parallel, ParallelPrimitive
from dicomnode.lib.io import Directory, File, fill_patient_storage_from_file_system
from dicomnode.lib.logging import queue_logger_thread_target, set_logger,\
  LoggerConfig, log_traceback

from dicomnode.server.input import AbstractInput
from dicomnode.config import DicomnodeConfig
from dicomnode.server.pipeline_storage import PipelineStorage
from dicomnode.server.patient_node import PatientNode
from dicomnode.server.maintenance import MaintenanceThread

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

    if config is None:
      config = DicomnodeConfig(
        STUDY_EXPIRATION_DAYS=self.study_expiration_days,
        PATIENT_IDENTIFIER_TAG=self.patient_identifier_tag,
        LAZY_STORAGE=self.lazy_storage,
        AE_TITLE=self.ae_title,
        IP=self.ip,
        PORT=self.port,
        REQUIRED_CALLED_AET=self.require_called_aet,
        ARCHIVE_DIRECTORY=self._data_directory,
        PROCESSING_DIRECTORY=self._processing_directory,
        RUN_FILE=File(self.run_file) if self.run_file else None,
        LOG_OUTPUT=str(self.log_output),
        LOG_WHEN=self.log_when,
        LOG_LEVEL=self.log_level,
        LOG_FORMAT=self.log_format
      )

    self.config = config

    self.identifier = DicomIdentifier(identifying_tag=self.patient_identifier_tag)

    self.data_state = PipelineStorage(
      self.input,
      self.identifier,
      self.config
    )

    fill_patient_storage_from_file_system(
      config.ARCHIVE_DIRECTORY,
      self.data_state
    )

    self.is_open = False

    # Server validations and creation.
    self.dicom_application_entry = ApplicationEntity(ae_title = self.ae_title)
    self.server_thread: ThreadedAssociationServer | None = None
    # You need VerificationPresentationContexts for ECHOSCU
    # and you want ECHO-SCU
    contexts = VerificationPresentationContexts + self.supported_contexts
    self.dicom_application_entry.supported_contexts = contexts

    self.dicom_application_entry.require_called_aet = self.require_called_aet
    self.dicom_application_entry.require_calling_aet = self.require_calling_aet


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


  def _handle_c_store(self, event: evt.Event) -> int:
    dataset = dataset_from_event(event)

    try:
      if not self.filter(dataset):
        self.logger.info("Node rejected dataset: Received Dataset did not pass filter")
        return 0xB006 # Element discarded
    except Exception as exception:
      log_traceback(self.logger, exception, "User defined function filter produced an exception")
      return 0xA801

    try:
      self.data_state.add_image(dataset, id(event.assoc))
    except InvalidDataset:
      self.logger.info(f"Node rejected dataset: Received dataset doesn't have patient Identifier tag: {hex(self.patient_identifier_tag)}")
      return 0xB007
    except Exception as exception:
      log_traceback(self.logger, exception, "Adding Image to input produced an exception")
      return 0xA801

    return 0x0000


  #region handle_association_released
  def _handle_association_released(self, event: evt.Event):
    """This function is called whenever an association is released. Note that
    this function should not be used for processing, because there's still a
    connection to the peer.

    """
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Released.")

    return


  def _handle_connection_closed(self, event: evt.Event):
    """This function has the responsibility of starting trigger the processing.

    Args:
        event (evt.Event): _description_
    """
    self.logger.debug(f"Connection {event.address[0]} closed a connection") #type: ignore
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Closed a connection.")
    input_containers, failed_datasets = self.data_state.extract_input_container(id(event.assoc))


    if len(failed_datasets):
      dicom_collection =  display_dicom_collection(failed_datasets)
      self.logger.info(f"The association with {event.assoc.requestor.ae_title} send {dicom_collection} which was rejected")

    self._process_output(input_containers)

  def _process_output(self, input_containers: List[Tuple[str, PatientNode]],):
    """This function exists to the target for the queued pipeline"""

    for patient_id,patient_node in input_containers:
      input_container = patient_node.grind()

      args = ProcessRunnerArgs(
        patient_id=patient_id,
        input_container=input_container,
        log_config=self.queue_logging_config(),
        process_path=self.get_processing_directory_path(patient_id)
      )

      parallel_primitive = ParallelPrimitive.THREAD if self.processing_directory is None else ParallelPrimitive.PROCESS
      primitive = Parallel(parallel_primitive, self.Processor, args)
      primitive.join()

      if primitive.is_successful():
        self.logger.info(f"process {patient_id} successful, Removing {patient_id}'s images")
        patient_node.clean_up()
      else:
        self.logger.error(f"Failed to process {patient_id}!")



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


class AbstractQueuedPipeline(AbstractPipeline):
  """A pipeline that processes each object one at a time

  This might be very relevant when processing require a resource, such as GPU
  """
  process_queue: Queue[List[Tuple[str, PatientNode]]]

  queue_timeout = 0.05

  def process_worker(self):
    """Worker function for the process_queue thread"""
    while self.running:
      try:
        input_container = self.process_queue.get(timeout=self.queue_timeout)
        self.logger.info(f"Process queue got node {input_container}, "
                         f"approximate queue length: {self.process_queue.qsize()}")
        try:
          self._process_output(input_container)
        except Exception as exception:
          log_traceback(self.logger, exception, "Process worker encountered exception")
        finally:
          self.process_queue.task_done()
      except Empty:
        pass

  def _handle_association_closed(self, event: evt.Event):
    input_containers, failed_datasets = self.data_state.extract_input_container(id(event.assoc))

    if len(failed_datasets):
      dicom_collection = display_dicom_collection(failed_datasets)
      self.logger.info(f"The association {id(event.assoc)} with {event.assoc.requestor.ae_title}  send {dicom_collection} which was rejected")

    if input_containers:
      self.process_queue.put(input_containers)
    else:
      self.logger.info(f"The association {id(event.assoc)} with {event.assoc.requestor.ae_title} - didn't produce any InputContainers to be processed")


  def node_signal_handler_SIGINT(self, signal_, frame):
    self.running = False
    super().node_signal_handler_SIGINT(signal_, frame)

  def __init__(self) -> None:
    self.running = True

    # Super is called at the end of the function
    super().__init__()
    self._evt_handlers[evt.EVT_CONN_CLOSE] = self._handle_association_closed

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
