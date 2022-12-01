"""This module contains basic DIMSE messages
  """

__author__ = "Christoffer Vilstrup Jensen"

from dataclasses import dataclass
from enum import Enum

import logging
from typing import Iterable, Callable, Optional

from threading import Thread

from pydicom import Dataset
from pydicom.uid import UID
from pynetdicom.ae import ApplicationEntity
from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelMove

from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage, InvalidQueryDataset
from dicomnode.lib.dicom import make_meta

logger = logging.getLogger("dicomnode")

class QueryLevels(Enum):
  PATIENT="PATIENT"
  STUDY="STUDY"
  SERIES="SERIES"


@dataclass
class Address:
  ip: str
  port: int
  ae_title: str # SCP ae title

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
                error_callback_func: Optional[Callable[[Address, Dataset, Dataset], None]] = None
  ):
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
          error_callback_func(Address, response, dataset)
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
    Could

  """
  if "QueryRetrieveLevel" not in dataset:
    dataset.QueryRetrieveLevel = query_level

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
    address.ae_title
  )

  successful_send = True
  if assoc.is_established:
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
      SCU AE:{SCU_AE}
    """
    logger.error(error_message)
    successful_send = False

  if not successful_send:
    raise CouldNotCompleteDIMSEMessage

def send_move_daemon(SCU_AE: str,
                     address : Address,
                     dataset : Dataset,
                     query_level: QueryLevels= QueryLevels.PATIENT
  ) -> Thread:
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
  daemon = Thread(target=send_move, daemon=True, args=(SCU_AE, address, dataset), kwargs={'query_level' : query_level})
  daemon.run()
  return daemon
