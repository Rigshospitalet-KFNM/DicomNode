"""This module contains basic DIMSE messages
  """

__author__ = "Christoffer Vilstrup Jensen"

from dataclasses import dataclass
from enum import Enum

import logging
from typing import Iterable, Callable, Optional

from pydicom import Dataset
from pydicom.uid import UID
from pynetdicom.ae import ApplicationEntity
from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelMove

from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage, InvalidQueryDataset


logger = logging.getLogger("dicomnode")

class QueryLevels(Enum):
  PATIENT : "PATIENT"
  STUDY   : "STUDY"
  SERIES  : "SERIES"


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

def send_image(SCU_AE: str, address: Address, dicom_image: Dataset ) -> Dataset:
  ae = ApplicationEntity(ae_title=SCU_AE)
  ae.add_requested_context(dicom_image.SOPClassUID)
  assoc = ae.associate(
    address.ip,
    address.port,
    ae_title=address.ae_title
  )
  if assoc.is_established:
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

def send_images(SCU_AE: str, address: Address, dicom_images: Iterable[Dataset], error_callback_func: Optional[Callable[[], None]] = None):
  ae = ApplicationEntity(ae_title=SCU_AE)
  ae.add_requested_context(dicom_images[0].SOPClassUID)
  assoc = ae.associate(
    address.ip,
    address.port,
    ae_title=address.ae_title
  )
  if assoc.is_established:
    for dataset in dicom_images:
      response = assoc.send_c_store(dataset)
      if(response.status != 0x0000) and error_callback_func is not None:
        assoc.release()
        raise CouldNotCompleteDIMSEMessage
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

def send_move(SCU_AE: str, address : Address, dataset : Dataset, query_level: QueryLevels = QueryLevels.PATIENT):
  """This function sends 
  

  """
  query_request_context = PatientRootQueryRetrieveInformationModelMove

  ae = ApplicationEntity(SCU_AE)
  ae.add_requested_context(query_request_context)
  
  if "QueryRetrieveLevel" not in dataset:
    dataset.QueryRetrieveLevel = query_level

  if query_level == QueryLevels.PATIENT and 'PatientID' not in dataset:
    raise InvalidQueryDataset

  if query_level == QueryLevels.STUDY and 'StudyInstanceUID' not in dataset:
    raise InvalidQueryDataset

  if query_level == QueryLevels.SERIES and 'SeriesInstanceUID' not in dataset:
    raise InvalidQueryDataset

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
        logger.debug("Failed to send")
        logger.debug(f"status: {status}")
        logger.debug(f"identifier: {identifier}")
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
