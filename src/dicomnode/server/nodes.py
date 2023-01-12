"""This module is the base class for all pipelines

  The life cycle of the pipeline is:

 - Storage Association is created
 - A number of C-stores is send to the pipeline.
 - The Association is released
 - For each updated patientTree, checks if all InputDataClasses validates
 - For each patient where they do, process them and dispatch them

"""

__author__ = "Christoffer Vilstrup Jensen"

from pynetdicom import evt
from pynetdicom.ae import ApplicationEntity as AE
from pynetdicom.presentation import AllStoragePresentationContexts, PresentationContext
from pydicom import Dataset

from dicomnode.lib.exceptions import InvalidDataset, CouldNotCompleteDIMSEMessage, IncorrectlyConfigured
from dicomnode.lib.dicomFactory import Blueprint, DicomFactory, FillingStrategy
from dicomnode.server.input import AbstractInput
from dicomnode.server.pipelineTree import PipelineTree, InputContainer, PatientNode
from dicomnode.server.output import PipelineOutput, NoOutput


import logging
import traceback
from threading import Thread
from queue import Queue

from copy import deepcopy
from logging import StreamHandler, getLogger
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from sys import stdout
from typing import Dict, Type, List, Optional, Set, Any, NoReturn, Union, Tuple


correct_date_format = "%Y/%m/%d %H:%M:%S"

class AbstractPipeline():
  """Base Class for an image processing Pipeline
  Creates a SCP server on object instantiation unless passed start=False

  Should be subclassed with your implementation of the process function at least.

  Check tutorials/ConfigurationOverview.md for an overview of attributes
  """

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
  filling_strategy: Optional[FillingStrategy] = None
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
  log_path: Optional[Union[str, Path]] = None
  log_level: int = logging.INFO
  log_format: str = "%(asctime)s %(name)s %(levelname)s %(message)s"
  disable_pynetdicom_logger: bool = True

  def close(self) -> None:
    """Closes all connections active connections.
      If your application includes additional connections, you should overwrite this method,
      And close any connections and call the super function.
    """
    self.ae.shutdown()

  def open(self, blocking=True) -> Optional[NoReturn]:
    """Opens all connections active connections.
      If your application includes additional connections, you should overwrite this method,
      And open any connections and call the super function.

      Keyword Args:
        blocking (bool) : if true, this functions doesn't return.
    """
    self.ae.start_server(
      (self.ip,self.port),
      block=blocking,
      evt_handlers=self._evt_handlers)

  def __init__(self, start=True) -> Optional[NoReturn]: #type: ignore
    # logging
    if self.log_path:
      logging.basicConfig(
        level=self.log_level,
        format=self.log_format,
        datefmt=correct_date_format,
        handlers=[TimedRotatingFileHandler(
          filename=self.log_path,
          when='W0',
          backupCount=self.backup_weeks,
        )]
      )
    else:
      logging.basicConfig(
        level=self.log_level,
        format=self.log_format,
        datefmt=correct_date_format,
        handlers=[StreamHandler(
          stream=stdout
        )]
      )
    self.logger = getLogger("dicomnode")
    if self.disable_pynetdicom_logger:
      getLogger("pynetdicom").setLevel(logging.CRITICAL + 1)

    # Handler setup
    # class needs to be instantiated before handlers can be defined
    self._evt_handlers = [
      (evt.EVT_C_STORE,  self._handle_store),
      (evt.EVT_ACCEPTED, self._association_accepted),
      (evt.EVT_RELEASED, self._association_released)
    ]
    self.updated_patients: Dict[Optional[int], Set] = {

    }

    # Load state
    if self.data_directory is not None:
      if not isinstance(self.data_directory, Path):
        self.logger.warn("root_data_directory is not of type Path, attempting to convert!")
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

    # Validates that Pipeline is configured correctly.
    self.ae = AE(ae_title = self.ae_title)
    self.ae.supported_contexts = self.supported_contexts

    self.ae.require_called_aet = self.require_called_aet
    self.ae.require_calling_aet = self.require_calling_aet

    self.post_init(start=start)
    if start:
      self.open() #pragma: no cover

  def _log_user_error(self, Exp: Exception, user_function: str):
    self.logger.critical(f"Encountered error in user function {user_function}")
    self.logger.critical(f"The exception type: {Exp.__class__.__name__}")

  def _handle_store(self, event: evt.Event) -> int:
    dataset = event.dataset
    try:
      if not self.filter(dataset):
        self.logger.warning("Dataset discarded")
        return 0xB006 # Element discarded
    except Exception as E:
      self._log_user_error(E, "Filter")
      return 0xA801

    if self.patient_identifier_tag in dataset:
      patientID = deepcopy(dataset[self.patient_identifier_tag].value)
      try:
        self.data_state.add_image(dataset)
        self.updated_patients[event.assoc.native_id].add(patientID)
      except InvalidDataset:
        self.logger.debug(f"Received dataset is not accepted by any inputs")
        return 0xB006
    else:
      self.logger.debug(f"Node: Received dataset, doesn't have patient Identifier tag")
      return 0xB007

    return 0x0000

  def _association_accepted(self, event: evt.Event):
    self.logger.debug(f"Association with {event.assoc.requestor.ae_title} - {event.assoc.requestor.address} Accepted")
    for requested_context in event.assoc.requestor.requested_contexts:
      if requested_context.abstract_syntax is not None:
        if requested_context.abstract_syntax.startswith("1.2.840.10008.5.1.4.1.1"):
          self.updated_patients[event.assoc.native_id] = set()
      else:
        self.logger.error("Requestor have no abstract syntax? this is impossible") # pragma: no cover Unreachable code

  def _association_released(self, event: evt.Event):
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Released.")
    for patient_ID in self.updated_patients[event.assoc.native_id]:
      if (PatientData := self.data_state.validate_patient_ID(patient_ID)) is not None:
        try:
          result = self.process(PatientData)
        except Exception as E:
          self._log_user_error(E, "Process")
        else:
          if self.dispatch(result):
            self.logger.debug(f"Removing Patient {patient_ID}")
            self.data_state.remove_patient(patient_ID)
          else:
            self.logger.error("Unable to send output")
      # Failure Logging is done by validate_patient_ID
    del self.updated_patients[event.assoc.native_id] # Removing updated Patients

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

  def post_init(self, start : bool) -> None:
    """This function is called just before the server is started.
      The idea being that a user change this function to run some arbitrary code before the Dicom node starts.
      This would often be

    Args:
        start (bool): Indication if the server should start
    """
    pass


class AbstractQueuedPipeline(AbstractPipeline):
  """A pipeline that processes each object one at a time
  """
  process_queue: Queue[Tuple[str, InputContainer]]
  dispatch_queue: Queue[Tuple[str,PipelineOutput]]

  def process_worker(self):
    """Worker function for the process_queue"""
    while True:
      PatientID, input_container = self.process_queue.get()
      try:
        output = self.process(input_container)
      except Exception as E:
        self._log_user_error(E, "process")
      else:
        self.dispatch_queue.put((PatientID, output))
      finally:
        self.process_queue.task_done()

  def dispatch_worker(self):
    while True:
      PatientID, output = self.dispatch_queue.get()
      success = self.dispatch(output)
      if success:
        self.data_state.remove_patient(PatientID)
      else:
        self.logger.error(f"Could not export data")

      self.dispatch_queue.task_done()

  def _association_released(self, event: evt.Event):
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Released.")
    for patient_ID in self.updated_patients[event.assoc.native_id]:
      if (input_container := self.data_state.validate_patient_ID(patient_ID)) is not None:
        self.logger.debug(f"Sufficient data - Calling Processing")
        self.process_queue.put((patient_ID, input_container))
    del self.updated_patients[event.assoc.native_id]

  def __init__(self, start=True) -> Optional[NoReturn]:
    self.process_queue = Queue()
    self.dispatch_queue = Queue()

    self.process_thread = Thread(target=self.process_worker, daemon=True)
    self.dispatch_thread = Thread(target=self.dispatch_worker, daemon=True)

    self.process_thread.start()
    self.dispatch_thread.start()



    super().__init__(start)

  def close(self) -> None:
    self.process_queue.join()
    self.dispatch_queue.join()

    return super().close()

class AbstractThreadedPipeline(AbstractPipeline):
  """Pipeline that creates threads to handle storing, to minimize IO load
  """
  threads: Dict[Optional[int],List[Thread]] = {}

  def _handle_store(self, event: evt.Event) -> int:
    thread: Thread = Thread(target=super()._handle_store, args=[event], daemon=True)
    thread.start()
    if event.assoc.native_id in self.threads:
      self.threads[event.assoc.native_id].append(thread)
    else:
      self.threads[event.assoc.native_id] = [thread]
    return 0x0000

  def _join_threads(self, assoc_name:Optional[int] = None) -> None:
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

  def _association_released(self, event: evt.Event):
    self._join_threads(event.assoc.native_id)
    return super()._association_released(event)
