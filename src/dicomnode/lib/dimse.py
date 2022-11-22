"""This module contains basic DIMSE messages
  """

__author__ = "Christoffer Vilstrup Jensen"

from dataclasses import dataclass
import logging
from typing import Iterable, Callable, Optional

from pydicom import Dataset
from pydicom.uid import UID
from pynetdicom.ae import ApplicationEntity

from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage


logger = logging.getLogger("dicomnode")

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
