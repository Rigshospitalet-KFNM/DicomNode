"""This module contains basic DIMSE messages
  """

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Callable, Optional

# Third party packages
from pydicom import Dataset
from pydicom.uid import UID
from pynetdicom.ae import ApplicationEntity
from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelMove # type: ignore

# Dicomnode packages
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage, InvalidQueryDataset
from dicomnode.dicom import make_meta
from dicomnode.lib.utils import ThreadWithReturnValue
from dicomnode.lib.logging import get_logger


logger = get_logger()

class QueryLevels(Enum):
  PATIENT="PATIENT"
  STUDY="STUDY"
  SERIES="SERIES"

class DIMSE_StatusCodes(Enum):
  SUCCESS = 0x0000
  PENDING = 0xFF00
  PENDING_MISSING_KEYS = 0xFF01
  CANCELLED = 0xFE00
  SOP_NOT_SUPPORTED = 0x0122
  OUT_OF_RESOURCES  =0xA700
  FAILED = 0xC000 # Note that this is a Range from 0xC000 to 0xCFFF



@dataclass
class Address:
  ip: str
  port: int
  ae_title: str # SCP ae title

  def __str__(self):
    return f"{self.ip}:{self.port} - {self.ae_title}"

@dataclass(init=False)
class ResponseAddress(Address):
  """Dynamic address of target"""

  def __init__(self):
    pass

def send_image(SCU_AE: str, address: Address, dicom_image: Dataset) -> Dataset:
  ae = ApplicationEntity(ae_title=SCU_AE)
  ae.add_requested_context(dicom_image.SOPClassUID)
  assoc = ae.associate(
    address.ip,
    address.port,
    ae_title=address.ae_title
  )
  if assoc.is_established:
    if hasattr(dicom_image, 'file_meta'):
      make_meta(dicom_image)
    if 0x00020010 not in dicom_image.file_meta:
      make_meta(dicom_image)
    response = assoc.send_c_store(dicom_image)
    assoc.release()
    return response
  else:
    error_message = f"""Could not connect to the SCP with the following inputs:
      IP: {address.ip}
      Port: {address.port}
      SCP AE: {address.ae_title}
      SCU AE: {SCU_AE}
    """
    logger.error(error_message)
    raise CouldNotCompleteDIMSEMessage("Could not connect")

def send_images(SCU_AE: str,
                address: Address,
                dicom_images: Iterable[Dataset],
                error_callback_func: Optional[Callable[[Address, Dataset, Dataset], None]] = None,
                logger=None
  ):
  if logger is None:
    logger = get_logger()
  ae = ApplicationEntity(ae_title=SCU_AE)
  contexts = set()
  for image in dicom_images:
    if image.SOPClassUID.name not in contexts:
      ae.add_requested_context(image.SOPClassUID)
      contexts.add(image.SOPClassUID.name)

  assoc = ae.associate(
    address.ip,
    address.port,
    ae_title=address.ae_title
  )
  if assoc.is_established:
    for dataset in dicom_images:
      if not hasattr(dataset, 'file_meta'):
        make_meta(dataset)
      if 0x00020010 not in dataset.file_meta:
        make_meta(dataset)
      response = assoc.send_c_store(dataset)
      if(response.Status != 0x0000):
        if error_callback_func is None:
          assoc.release()
          error_message = f"Could not send {dataset}\n Received Response: {response}"
          logger.error(error_message)
          raise CouldNotCompleteDIMSEMessage(f"Could not send {dataset}")
        else:
          error_callback_func(address, response, dataset)
    assoc.release()
  else:
    error_message = f"""Could not connect to the SCP with the following inputs:
      IP: {address.ip}
      Port: {address.port}
      SCP AE: {address.ae_title}
      SCU AE:{SCU_AE}
    """
    logger.error(error_message)
    raise CouldNotCompleteDIMSEMessage("Could not connect")
  return 0x0000

def send_images_thread(
    SCU_AE: str,
    address : Address,
    dicom_images: Iterable[Dataset],
    error_callback_func: Optional[Callable[[Address, Dataset, Dataset], None]] = None,
    daemon: bool = True) -> ThreadWithReturnValue:
  thread = ThreadWithReturnValue(group= None, target=send_images, args=[SCU_AE, address, dicom_images, error_callback_func], daemon=daemon)
  thread.start()
  return thread

def send_move(SCU_AE: str,
              address : Address,
              dataset : Dataset,
              query_level: QueryLevels = QueryLevels.PATIENT
  ) -> None:
  """This function sends a C-move to the address, as the SCU_AE to the SCU_AE

  Args:
    SCU_AE (str):
    address (Address):
    dataset (dataset):

  Kwargs:
    query_level (QueryLevel, optional):

  Raises:
    InvalidQueryDataset:
  """
  if "QueryRetrieveLevel" not in dataset:
    dataset.QueryRetrieveLevel = query_level.value

  if query_level == QueryLevels.PATIENT and 'PatientID' not in dataset:
    logger.error("Attempted to send a move at Patient level without a PatientID tag")
    raise InvalidQueryDataset

  if query_level == QueryLevels.STUDY and 'StudyInstanceUID' not in dataset:
    logger.error("Attempted to send a move at Study level without a StudyInstanceUID tag")
    raise InvalidQueryDataset

  if query_level == QueryLevels.SERIES and 'SeriesInstanceUID' not in dataset:
    logger.error("Attempted to send a move at Series level without a SeriesInstanceUID tag")
    raise InvalidQueryDataset

  query_request_context = PatientRootQueryRetrieveInformationModelMove
  ae = ApplicationEntity(SCU_AE)
  ae.add_requested_context(query_request_context)
  assoc = ae.associate(
    address.ip,
    address.port,
    ae_title=address.ae_title
  )

  successful_send = True
  if assoc.is_established:
    logger.debug("Sending C move")
    response = assoc.send_c_move(dataset, SCU_AE, query_request_context)
    for (status, identifier) in response:
      if status:
        logger.debug(f"status: {status}")
        logger.debug(f"identifier: {identifier}")
      else:
        logger.error("Failed to complete C-Move")
        logger.error(f"status: {status}")
        logger.error(f"identifier: {identifier}")
        successful_send = False
    assoc.release()
  else:
    error_message = f"""Could not connect to the SCP with the following inputs:
      IP: {address.ip}
      Port: {address.port}
      SCP AE: {address.ae_title}
      SCU AE: {SCU_AE}
    """
    logger.error(error_message)
    successful_send = False

  if not successful_send:
    raise CouldNotCompleteDIMSEMessage
  return None

def send_move_thread(SCU_AE: str,
                     address : Address,
                     dataset : Dataset,
                     query_level: QueryLevels= QueryLevels.PATIENT,
                     daemon: bool = True
  ) -> ThreadWithReturnValue:
  """Creates a thread, that sends a C-Move to the target.

  The main reason you want to use this function over the standard send C-Move
  is you don't want to wait for the result to terminate.

  Args:
      SCU_AE (str): The AE
      address (Address): _description_
      dataset (Dataset): _description_
      query_level (QueryLevels, optional): _description_. Defaults to QueryLevels.PATIENT.

  Returns:
      Thread: _description_
  """
  thread = ThreadWithReturnValue(target=send_move, daemon=daemon, args=(SCU_AE, address, dataset), kwargs={'query_level' : query_level})
  thread.start()
  return thread
