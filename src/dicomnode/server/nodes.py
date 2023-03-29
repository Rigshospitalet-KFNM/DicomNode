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
from datetime import datetime, timedelta
import logging
from logging import getLogger
from os import chdir, getcwd
from pathlib import Path
from queue import Queue, Empty
import shutil
from sys import stdout
from threading import Thread
from time import sleep
import traceback
from typing import Dict, Type, List, Optional, Set, Any, NoReturn, Union, Tuple, TextIO

# Third part packages
from pynetdicom import evt
from pynetdicom.ae import ApplicationEntity as AE
from pynetdicom.presentation import AllStoragePresentationContexts, PresentationContext, VerificationPresentationContexts
from pydicom import Dataset

# Dicomnode packages
from dicomnode.lib.io import TemporaryWorkingDirectory
from dicomnode.lib.exceptions import InvalidDataset, CouldNotCompleteDIMSEMessage, IncorrectlyConfigured
from dicomnode.lib.dicom_factory import Blueprint, DicomFactory, FillingStrategy
from dicomnode.lib.logging import log_traceback, get_logger, set_logger
from dicomnode.server.input import AbstractInput
from dicomnode.server.pipelineTree import PipelineTree, InputContainer, PatientNode
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

  # AE configuration tags
  ae_title: str = "Your_AE_TITLE"
  ip: str = 'localhost'
  port: int = 104
  supported_contexts: List[PresentationContext] = AllStoragePresentationContexts
  require_called_aet: bool = True
  require_calling_aet: List[str] = []


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
    # 4. If start is true, start the server

    # logging
    if isinstance(self.log_output, str):
      self.log_output = Path(self.log_output)

    self.logger = get_logger()
    """
    self.logger = set_logger(
      log_output=self.log_output,
      log_level=self.log_level,
      format=self.log_format,
      date_format=self.log_date_format,
      backupCount=self.backup_weeks
    )
    """

    if self.disable_pynetdicom_logger:
      getLogger("pynetdicom").setLevel(logging.CRITICAL + 1)

    self.maintenance_thread = Thread(target=self.maintenance_worker, daemon=True)
    self.maintenance_thread.start()

    # Load any previous state
    if self.data_directory is not None:
      if not isinstance(self.data_directory, Path):
        self.data_directory = Path(self.data_directory)

      if self.data_directory.is_file():
        raise IncorrectlyConfigured("The root data directory exists as a file.")

      if not self.data_directory.exists():
        self.data_directory.mkdir(parents=True)

    options = self.pipelineTreeType.Options(
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
      options
    )

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
      (evt.EVT_C_STORE,  self.handle_C_STORE_message),
      (evt.EVT_ACCEPTED, self.association_accepted),
      (evt.EVT_RELEASED, self.association_released)
    ]
    self.updated_patients: Dict[Optional[int], Set] = {}

    self.post_init()

  def _log_user_error(self, Exp: Exception, user_function: str):
    self.logger.critical(f"Encountered error in user function {user_function}")
    self.logger.critical(f"The exception type: {Exp.__class__.__name__}")
    exception_info = traceback.format_exc()
    self.logger.critical(exception_info)

  def handle_C_STORE_message(self, event: evt.Event) -> int:
    dataset = event.dataset
    dataset.file_meta = event.file_meta
    try:
      if not self.filter(dataset):
        self.logger.warning("Dataset discarded")
        return 0xB006 # Element discarded
    except Exception as E:
      self._log_user_error(E, "Filter")
      return 0xA801

    if event.assoc.native_id is None:
      raise IncorrectlyConfigured # pragma no cover # unreachable code

    return self.store_dataset(dataset, event.assoc.native_id)

  def store_dataset(self, dataset: Dataset, assoc_id: int) -> int:
    if self.patient_identifier_tag in dataset:
      patientID = deepcopy(dataset[self.patient_identifier_tag].value)
      try:
        self.data_state.add_image(dataset)
        self.updated_patients[assoc_id].add(patientID)
      except InvalidDataset:
        self.logger.debug(f"Received dataset is not accepted by any inputs")
        return 0xB006
    else:
      self.logger.debug(f"Node: Received dataset, doesn't have patient Identifier tag")
      return 0xB007

    return 0x0000


  def association_accepted(self, event: evt.Event):
    self.logger.debug(f"Association with {event.assoc.requestor.ae_title} - {event.assoc.requestor.address} Accepted")
    for requested_context in event.assoc.requestor.requested_contexts:
      if requested_context.abstract_syntax is not None:
        if requested_context.abstract_syntax.startswith("1.2.840.10008.5.1.4.1.1"):
          self.updated_patients[event.assoc.native_id] = set()
      else:
        self.logger.error("Requestor have no abstract syntax? this is impossible") # pragma: no cover Unreachable code

  def association_released(self, event: evt.Event):
    """This function is called whenever an association is released
    It's the controller function for processing data

    Args:
        event (evt.Event): _description_
    """
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Released.")
    for patient_ID in self.updated_patients[event.assoc.native_id]:
      self.initial_environment_for_processing_patient(patient_ID)
    del self.updated_patients[event.assoc.native_id] # Removing updated Patients


  def initial_environment_for_processing_patient(self, patient_ID: str) -> None:
    if self.data_state.validate_patient_ID(patient_ID):
      self.logger.debug(f"Sufficient data for patient {patient_ID}")
      if self.processing_directory is not None:
        with TemporaryWorkingDirectory(self.processing_directory / str(patient_ID)) as twd:
          self.main_processing_loop(patient_ID)
      else:
        self.main_processing_loop(patient_ID)

    else:
      self.logger.debug(f"insufficient data for patient {patient_ID}")

  def main_processing_loop(self, patient_ID):
    try:
      patient_input_container = self.data_state.get_patient_input_container(patient_ID)
      result = self.process(patient_input_container)
    except Exception as E:
      self._log_user_error(E, "Process")
      raise E
    else:
      self.logger.debug(f"Process {patient_ID} Successful, Dispatching output!")
      if self.dispatch(result):
        self.logger.debug("Dispatching Successful")
        self.logger.debug(f"Removing Patient {patient_ID}")
        self.data_state.remove_patient(patient_ID)
      else:
        self.logger.error("Unable to dispatch pipeline output")



  def close(self) -> None:
    """Closes all connections active connections & cleans up any temporary working spaces.

      If your application includes additional connections, you should overwrite this method,
      And close any connections and call the super function.
    """
    if self.processing_directory is not None:
      chdir(self.__cwd)
      shutil.rmtree(self.processing_directory)

    self.logger.info("Closing Server!")

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

    self.logger.info(f"Starting Server at port: {self.port} and AE: {self.ae_title}")

    self.ae.start_server(
      (self.ip,self.port),
      block=blocking,
      evt_handlers=self._evt_handlers)

  def maintenance_worker(self) -> NoReturn: #pragma no cover
    """This is the controller for the worker thread

    Should run the clean up function every midnight
    """
    while True:
      sleep(self.calculate_time_to_next_maintenance())
      self.maintenance()


  def calculate_time_to_next_maintenance(self) -> float:
    """Calculates the time in seconds to the next scheduled clean up"""
    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    clean_up_datetime = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0,0,0,0, tzinfo=now.tzinfo)
    time_delta = clean_up_datetime - now
    # Days are ignored in the time delta due function constraint of 1 run per day.
    return float(time_delta.seconds) # I guess you could add micro seconds here but WHO CARES

  def maintenance(self, now = datetime.now()) -> None:
    """Removes old studies in the pipeline to ensure GDPR compliance
    """
    # Note this might cause some bug, where a patient is being processed, and at the same time removed
    # This is considered so unlikely, that it's a bug I accept in the code
    expiry_datetime = now - timedelta(days=self.study_expiration_days)
    self.data_state.remove_expired_studies(expiry_datetime)



  def dispatch(self, output: PipelineOutput) -> bool:
    """This function is responsible for triggering exporting of data and handling errors.
      You should consider if it's possible to create your own output rather than overwriting this function

      Args:
        output: PipelineOutput
      Returns:
        bool - If the output was successful in exporting the data.
    """
    try:
      success = output.send()
    except Exception as E:
      self._log_user_error(E, "Output sending")
      success = False
    return success


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


class AbstractQueuedPipeline(AbstractPipeline):
  """A pipeline that processes each object one at a time

  This might be very relevant when processing require a resource, such as GPU
  """
  process_queue: Queue[Tuple[str, InputContainer]]
  dispatch_queue: Queue[Tuple[str, PipelineOutput]]
  queue_timeout = 0.05

  def process_worker(self):
    """Worker function for the process_queue"""
    while self.running:
      try:
        PatientID, input_container = self.process_queue.get(timeout=self.queue_timeout)
        self.logger.info(f"Process Queue extracted: {PatientID}")
        try:
          output = self.process(input_container)
        except Exception as E:
          self._log_user_error(E, "process")
        else:
          self.logger.info("Successfully Processed Task")
          try:
            success = self.dispatch(output)
          except Exception as exception:
            log_traceback(self.logger, exception, "Failed to process dispatch")
          else:
            if success:
              self.data_state.remove_patient(PatientID)
            else:
              self.logger.error("could not export data!")
        finally:
          self.logger.info("Finished Handel Task")
          self.process_queue.task_done()
      except Empty as E:
        pass

  def association_released(self, event: evt.Event):
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Released.")
    for patient_ID in self.updated_patients[event.assoc.native_id]:
      if self.data_state.validate_patient_ID(patient_ID):
        self.logger.debug(f"Sufficient data - Calling Processing")
        self.process_queue.put((patient_ID, self.data_state.get_patient_input_container(patient_ID)))
    del self.updated_patients[event.assoc.native_id]

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

  def handle_C_STORE_message(self, event: evt.Event) -> int:
    thread: Thread = Thread(target=super().handle_C_STORE_message, args=[event], daemon=False)
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

  def association_released(self, event: evt.Event):
    self.join_threads(event.assoc.native_id)
    return super().association_released(event)
