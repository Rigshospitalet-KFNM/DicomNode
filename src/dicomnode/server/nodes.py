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
from pynetdicom.presentation import AllStoragePresentationContexts
from pydicom import Dataset

from dicomnode.lib.dimse import Address, send_images
from dicomnode.lib.exceptions import InvalidDataset, CouldNotCompleteDIMSEMessage, IncorrectlyConfigured
from dicomnode.lib.dicomFactory import Blueprint, DicomFactory
from dicomnode.server.pipelineTree import PipelineTree, InputContainer
from dicomnode.server.input import AbstractInput


import logging
import traceback

from abc import ABC, abstractmethod

from gc import collect as garbage_collect

from copy import copy, deepcopy
from logging import StreamHandler, getLogger
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from sys import stdout, getrefcount
from typing import Dict, Type, List, Optional, MutableSet, Any, Iterable, NoReturn, Union


correct_date_format = "%Y/%m/%d %H:%M:%S"

class AbstractPipeline(ABC):
  """Abstract Class for a Pipeline, which acts a SCP

  Requires the following attributes before it can be instantiated.
    * ae_title : str
    * config_path : Union[str, PathLike]
    * process : Callable
  """

  # Input configuration
  input: Dict[str, Type[AbstractInput]] = {}
  input_config: Dict[str, Dict[str, Any]] = {}
  patient_identifier_tag: int = 0x00100020 # Patient ID
  root_data_directory: Optional[Path] = None
  dicom_factory: Optional[DicomFactory] = None
  pipelineTreeType: Type[PipelineTree] = PipelineTree
  inputContainerType: Type[InputContainer] = InputContainer

  # Output Configuration
  endpoints: List[Address] = []

  # AE configuration tags
  ae_title: str = "Your_AE_TITLE"
  ip: str = 'localhost'
  port: int = 104
  supported_contexts = AllStoragePresentationContexts
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

  def open(self, blocking=True) -> NoReturn: #type: ignore
    """Opens all connections active connections.
      If your application includes additional connections, you should overwrite this method,
      And open any connections and call the super function.

      Keyword Args:
        blocking (bool) : if true, this functions doesn't return.
    """
    self.ae.require_called_aet = self.require_called_aet
    self.ae.require_calling_aet = self.require_calling_aet

    self.ae.start_server(
      (self.ip,self.port),
      block=blocking,
      evt_handlers=self.__evt_handlers)

  def __init__(self, start=True) -> NoReturn: #type: ignore
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
    self.__evt_handlers = [
      (evt.EVT_C_STORE, self.__handle_store),
      (evt.EVT_ACCEPTED, self.__association_accepted),
      (evt.EVT_RELEASED, self.__association_released)
    ]
    self.__updated_patients: Dict[str, MutableSet] = {

    }

    # Load state
    if self.root_data_directory is not None:
      if not isinstance(self.root_data_directory, Path):
        self.logger.warn("root_data_directory is not of type Path, attempting to convert!")
        self.root_data_directory = Path(self.root_data_directory)

      if self.root_data_directory.is_file():
        raise IncorrectlyConfigured("The root data directory exists as a file.")

      if not self.root_data_directory.exists():
        self.root_data_directory.mkdir()


    options = self.pipelineTreeType.Options(
      input_container=self.inputContainerType,
      data_directory=self.root_data_directory,
      factory=self.dicom_factory
    )

    self.__data_state: PipelineTree = self.pipelineTreeType(
      self.patient_identifier_tag,
      self.input,
      options
    )

    # Validates that Pipeline is configured correctly.
    self.ae = AE(ae_title = self.ae_title)
    self.ae.supported_contexts = self.supported_contexts

    self.post_init(start=start)
    if start:
      self.open() #pragma: no cover

  def __log_user_error(self, Exp: Exception, user_function: str):
    self.logger.critical(f"Encountered error in user function {user_function}")
    self.logger.critical(f"The exception type: {Exp.__class__.__name__}")

  def __handle_store(self, event: evt.Event):
    dataset = event.dataset
    try:
      if not self.filter(dataset):
        self.logger.warning("Dataset discarded")
        return 0xB006 # Element discarded
    except Exception as E:
      self.__log_user_error(E, "Filter")
      return 0xA801

    if self.patient_identifier_tag in dataset:
      patientID = deepcopy(dataset[self.patient_identifier_tag].value)
      try:
        self.__data_state.add_image(dataset)
        self.__updated_patients[event.assoc.name].add(patientID)
      except InvalidDataset:
        self.logger.debug(f"Received dataset is not accepted by any inputs")
        return 0xB006
    else:
      self.logger.debug(f"Node: Received dataset, doesn't have patient Identifier tag")
      return 0xB007

    return 0x0000

  def __association_accepted(self, event: evt.Event):
    self.logger.debug(f"Association with {event.assoc.requestor.ae_title} - {event.assoc.requestor.address} Accepted")
    for requested_context in event.assoc.requestor.requested_contexts:
      if requested_context.abstract_syntax.startswith("1.2.840.10008.5.1.4.1.1"): #type: ignore There is an error here most likely.

        self.__updated_patients[event.assoc.name] = set()

  def __association_released(self, event: evt.Event):
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} Released.")
    for patient_ID in self.__updated_patients[event.assoc.name]:
      if (PatientData := self.__data_state.validate_patient_ID(patient_ID)) is not None:
        self.logger.debug(f"Sufficient data - Calling Processing")
        try:
          result = self.process(PatientData)
        except Exception as E:
          self.__log_user_error(E, "Process")
        else:
          if self.dispatch(result):
            self.logger.debug("Removing Patient")
            PatientData._cleanup()
            del self.__updated_patients[event.assoc.name]
          else:
            self.logger.error("Unable to send to addresses")
      else:
        self.logger.debug(f"Dataset was not valid for {patient_ID}")

  def dispatch(self, images: Iterable[Dataset]) -> bool:
    success: bool = True
    for address in self.endpoints:
      try:
        self.logger.debug(f"Sending datasets to {Address}")
        send_images(self.ae_title, address, images)
      except CouldNotCompleteDIMSEMessage:
        self.logger.error(f"Could not send response to {address}")
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

  @abstractmethod
  def process(self, input_data: InputContainer) -> Iterable[Dataset]:
    raise NotImplemented #pragma: no cover

  def post_init(self, start : bool) -> None:
    """This function is called just before the server is started.
      The idea being that a user change this function to run some arbitrary code before the Dicom node starts.
      This would often be

    Args:
        start (bool): Indicatation if the server should start

    """
    pass
