"""This module contains basic DIMSE messages
  """

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Callable, Optional

# Third party packages
from pydicom import Dataset, DataElement
from pydicom.datadict import tag_for_keyword, dictionary_VR
from pydicom.uid import UID
from pynetdicom.ae import ApplicationEntity
from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelMove, StudyRootQueryRetrieveInformationModelMove, SeriesRootQueryRetrieveInformationModelMove, PatientRootQueryRetrieveInformationModelFind, StudyRootQueryRetrieveInformationModelFind, SeriesRootQueryRetrieveInformationModelFind # type: ignore
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

  def find_sop_class(self) -> UID:
    mapping = {
      QueryLevels.PATIENT : PatientRootQueryRetrieveInformationModelFind,
      QueryLevels.STUDY : StudyRootQueryRetrieveInformationModelFind,
      QueryLevels.SERIES : SeriesRootQueryRetrieveInformationModelMove
    }

    return mapping[self]

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

class AssociationContextManager:
  def __init__(self, ae: ApplicationEntity, *args, **kwargs) -> None:
    self.ae = ae
    self.args = args
    self.kwargs = kwargs
    self.assoc = None

  def __enter__(self):
    self.assoc = self.ae.associate(
      *self.args,
      **self.kwargs
    )
    if self.assoc.is_established:
      return self.assoc
    else:
      return None

  def __exit__(self, exception_type, exception, traceback):
    if self.assoc is not None and self.assoc.is_established:
      self.assoc.release()


@dataclass(init=False)
class ResponseAddress(Address):
  """Dynamic address of target"""

  def __init__(self):
    pass

def send_image(SCU_AE: str, address: Address, dicom_image: Dataset) -> Dataset:
  """Sends a single dataset to a SCP

  Args:
      SCU_AE (str): The AE title of your program, that is recognized by the SCP
      address (Address): The Address of your destination
      dicom_image (Dataset): The Image that you sends

  Raises:
      CouldNotCompleteDIMSEMessage: Raised If unable to

  Returns:
      Dataset: _description_
  """
  ae = ApplicationEntity(ae_title=SCU_AE)
  ae.add_requested_context(dicom_image.SOPClassUID)
  with AssociationContextManager(ae, address.ip,
    address.port,
    ae_title=address.ae_title) as assoc:

    if assoc is not None:
      if hasattr(dicom_image, 'file_meta'):
        make_meta(dicom_image)
      if 0x00020010 not in dicom_image.file_meta:
        make_meta(dicom_image)
      response = assoc.send_c_store(dicom_image)
      return response
  raise CouldNotCompleteDIMSEMessage(f"Could not connect to {address}")


def send_images(SCU_AE: str,
                address: Address,
                dicom_images: Iterable[Dataset],
                error_callback_func: Optional[Callable[[Address, Dataset, Dataset], None]] = None,
                logger=None):
  """Sends multiple dataset to a SCP

  Args:
      SCU_AE (str): The AE title of your program, that is recognized by the SCP
      address (Address): The Address of your destination
      dicom_images (Iterable[Dataset]): The Images that you sends to the address
  Kwargs:
      error_callback_func - (Optional[Callable[Address, Dataset, Dataset], None]) -
        User defined error function, that is called if the storage fails. The
        arguments to the functions are the address that was being send to, the
        response dataset and finally the dataset that failed to be stored.
      logger (Optional[Logger]) - Logger that this function will log to,
                                  default to None, and then this functions logs
                                  to the logger returned by get_logger

  Raises:
      CouldNotCompleteDIMSEMessage: Raised If unable to send the datasets and
                                    error_callback_func is None
  """

  if logger is None:
    logger = get_logger()
  ae = ApplicationEntity(ae_title=SCU_AE)
  added_contexts = set()
  for image in dicom_images:
    if isinstance(image, DataElement):
      raise TypeError("You passed a single dataset, either use send_image or wrap it in a list: [dicom_images]")
    if not isinstance(image, Dataset):
      raise TypeError("The dicom_images iterator yielded a non Dataset element")

    if image.SOPClassUID.name not in added_contexts:
      ae.add_requested_context(image.SOPClassUID)
      added_contexts.add(image.SOPClassUID.name)

  assoc = ae.associate(
    address.ip,
    address.port,
    ae_title=address.ae_title,
  )
  if assoc.is_established:
    try:
      for dataset in dicom_images:
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
    finally:
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

def create_query_ae(ae_title: str, query_level: QueryLevels) -> ApplicationEntity:
  ae = ApplicationEntity(ae_title)

  return ae

def validate_query_dataset(dataset: Dataset):
  if "QueryRetrieveLevel" not in dataset:
    return False

  if dataset.QueryRetrieveLevel == QueryLevels.PATIENT and 'PatientID' not in dataset:
    logger.error("Attempted to send a move at Patient level without a PatientID tag")
    return False

  if dataset.QueryRetrieveLevel == QueryLevels.STUDY and 'StudyInstanceUID' not in dataset:
    logger.error("Attempted to send a move at Study level without a StudyInstanceUID tag")
    return False

  if dataset.QueryRetrieveLevel == QueryLevels.SERIES and 'SeriesInstanceUID' not in dataset:
    logger.error("Attempted to send a move at Series level without a SeriesInstanceUID tag")
    return False

  return True

def create_query_dataset(query_level=QueryLevels.STUDY, **kwargs):
  """Generates a dataset that can be send with a C-FIND or a C-MOVE

  Args:
      query_level (QueryLevels, optional): The query level for the query.
        Defaults to QueryLevels.STUDY.

  Raises:
      ValueError: If a keyword doesn't match a Tag
      InvalidQueryDataset: If the produced dataset isn't valid for a query

  Returns:
      Dataset: The dataset to be used for query:

  Example:
  >>> create_query_dataset(query_level="Patient", PatientID="patient_id")

  """
  dataset = Dataset()

  dataset.QueryRetrieveLevel = query_level.value

  # Patient Level
  dataset.PatientName = None
  dataset.PatientID = None
  dataset.PatientBirthDate = None

  # Study Level
  dataset.AccessionNumber = None
  dataset.StudyDate = None
  dataset.StudyDescription = None
  dataset.StudyID = None

  # Series Level
  dataset.SeriesDescription = None
  dataset.SeriesNumber = None

  for tag_name, value in kwargs.items():
    tag = tag_for_keyword(tag_name)

    if tag is None:
      raise ValueError(f"Keyword: {tag_name} is not a ")

    vr = dictionary_VR(tag)
    dataset[tag] = DataElement(tag, vr, value)

  if not validate_query_dataset(dataset):
    raise InvalidQueryDataset

  return dataset

def send_images_thread(
    SCU_AE: str,
    address : Address,
    dicom_images: Iterable[Dataset],
    error_callback_func: Optional[Callable[[Address, Dataset, Dataset], None]] = None,
    daemon: bool = True) -> ThreadWithReturnValue:
  """Creates a thread that calls send_images:

  look at send_images for docs.
  """
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
  if 'QueryRetrieveLevel' not in dataset:
    dataset.QueryRetrieveLevel = query_level.value

  if not validate_query_dataset(dataset):
    raise InvalidQueryDataset(f"Incoming Dataset is not valid")

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
