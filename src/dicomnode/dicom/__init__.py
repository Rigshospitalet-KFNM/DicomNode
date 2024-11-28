"""Library methods for manipulation of pydicom.dataset objects
"""

# Python Standard Library
from functools import reduce
from enum import Enum
from typing import Any, List, Callable, Optional, Tuple, Union

# Third Party Libraries
import numpy
from pydicom import DataElement
from pydicom.dataset import Dataset, validate_file_meta
from pydicom.tag import BaseTag
from pydicom.uid import UID, generate_uid, ImplicitVRLittleEndian, ExplicitVRBigEndian, ExplicitVRLittleEndian
from pydicom.valuerep import PersonName

# Dicomnode packages
from dicomnode.constants import DICOMNODE_IMPLEMENTATION_UID, DICOMNODE_IMPLEMENTATION_NAME, DICOMNODE_VERSION
from dicomnode.lib.exceptions import InvalidDataset

PRIVATIZATION_VERSION = 1

# Ensure Correct loading
#import dicomnode.math as __math


class Reserved_Tags(Enum):
  PRIVATE_TAG_NAMES = 0xFD
  PRIVATE_TAG_VRS = 0xFE
  PRIVATE_TAG_VM = 0xFF


def gen_uid() -> UID:
  """Generates a Unique identifier with Dicomnode's UID prefix

  Returns:
      UID: A Unique identifier with Dicomnode's prefix.
  """
  return generate_uid(prefix=DICOMNODE_IMPLEMENTATION_UID + '.')

def make_meta(dicom: Dataset) -> None:
  """Similar to fix_meta_info method, however UID are generated with dicomnodes prefix instead

  Args:
      dicom (Dataset): dicom dataset to be updated
  Raises:
      InvalidDataset: If meta header cannot be generated or is Transfer syntax is not supported
  """
  if not 0x00080016 in dicom:
    raise InvalidDataset("Cannot create meta header without SOPClassUID")
  if not 0x00080018 in dicom:
    dicom.SOPInstanceUID = gen_uid()

  dicom.ensure_file_meta()
  if 'FileMetaInformationVersion' not in dicom.file_meta:
    dicom.file_meta.FileMetaInformationVersion = b'\x00\x01'
  if 'ImplementationClassUID' not in dicom.file_meta:
    dicom.file_meta.ImplementationClassUID = DICOMNODE_IMPLEMENTATION_UID
  if 'ImplementationVersionName' not in dicom.file_meta:
    dicom.file_meta.ImplementationVersionName = f"{DICOMNODE_IMPLEMENTATION_NAME} {DICOMNODE_VERSION}"
  if 'MediaStorageSOPClassUID' not in dicom.file_meta:
    dicom.file_meta.MediaStorageSOPClassUID = dicom.SOPClassUID
  if 'MediaStorageSOPInstanceUID' not in dicom.file_meta:
    dicom.file_meta.MediaStorageSOPInstanceUID = dicom.SOPInstanceUID
  if hasattr(dicom, 'is_little_endian') and dicom.is_little_endian is not None:
    dicom.is_little_endian = None
  if hasattr(dicom , 'is_implicit_VR') and dicom.is_implicit_VR is not None:
    dicom.is_implicit_VR = None

  if 'TransferSyntaxUID' not in dicom.file_meta:
    dicom.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    # ImplicitVRBigEndian which is not supported pydicom
  elif dicom.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.2":
    raise InvalidDataset("ImplicitVRBigEndian is not support by dicomnode")

  validate_file_meta(dicom.file_meta)


def get_tag(Tag: int) -> Callable[[Dataset], Any]:
  def retFunc(Dataset):
    if Tag in Dataset:
      return Dataset[Tag]
    else:
      return None
  return retFunc

def __check_if_tag_is_creator(tag: BaseTag) -> bool:
  return not(tag & 0xFF00)

def reader_function_version_1(dataset: Dataset, data_element: DataElement):
  # This is too ensure that the reserved tags remains the same.
  class Reserved_Tags(Enum):
    PRIVATE_TAG_NAMES = 0xFD
    PRIVATE_TAG_VRS = 0xFE
    PRIVATE_TAG_VM = 0xFF


__reader_functions = {
  1 : reader_function_version_1
}

def get_dicomnode_creator_header():
  return f"Dicomnode - Private tags version: {PRIVATIZATION_VERSION}"


def __get_reader_function(data_element: DataElement) -> Optional[Callable[[Dataset, DataElement], None]]:
  stringified_value = str(data_element.value)
  if not stringified_value.startswith("Dicomnode - Private tags version: "):
    return None

  _, version_str = stringified_value.split("Dicomnode - Private tags version: ")

  return __reader_functions.get(int(version_str), None)


def refresh_dataset_dict(dataset: Dataset):
  for data_element in dataset:
    if data_element.is_private and __check_if_tag_is_creator(data_element.tag):
      read_function  = __get_reader_function(data_element)
      if read_function is not None:
        read_function(dataset, data_element)


def extrapolate_image_position_patient(
    slice_thickness:float,
    orientation: int,
    initial_position:  Tuple[float, float, float],
    image_orientation: Tuple[float, float, float, float, float, float],
    image_number: int,
    slices: int) -> List[List[float]]:
  """Extrapolates a list of image positions from an initial position.

  Useful for when you want to generate positions for an series.

  Assumes even slice thickness throughout the series

  Args:
      slice_thickness (float): Thickness of an image slice
      orientation (int): Direction to the extrapolation
      initial_position (Tuple[float, float, float]): Initial position as x,y,z
      image_orientation (Tuple[float, float, float, float, float, float]): Vectors defining the patient vector space
      image_number (int): Image number of the initial position
      slices (int): Number of slices in the extrapolated positions

  Returns:
      List[List[float]]: List of positions in [x,y,z] sub-lists
  """

  cross_vector = slice_thickness * orientation * numpy.array([
    image_orientation[1] * image_orientation[5] - image_orientation[2] * image_orientation[4],
    image_orientation[2] * image_orientation[3] - image_orientation[0] * image_orientation[5],
    image_orientation[0] * image_orientation[4] - image_orientation[1] * image_orientation[3],
  ])

  position = [numpy.array(initial_position) + (slice_num - image_number) * cross_vector for slice_num in numpy.arange(1,slices + 1, 1, dtype=numpy.float64)]

  return [[float(val) for val in pos] for pos in position]



def extrapolate_image_position_patient_dataset(dataset: Dataset, slices: int) -> List[List[float]]:
  """Wrapper function for extrapolate_image_position_patient
  Extracts values from a dataset and passes it to the function

  Args:
      dataset (Dataset): Dataset that contains:
        * 0x00180050 - SliceThickness
        * 0x00185100 - PatientPosition
        * 0x00200013 - InstanceNumber
        * 0x00200032 - ImagePositionPatient
        * 0x00200037 - ImageOrientation
      slices (int): Number of slices in extrapolation

  Raises:
      InvalidDataset: If the dataset is invalid

  Returns:
      List[List[float]]: List of positions in [x,y,z] sub-lists
  """
  required_tags = [
    0x00180050, # SliceThickness
    0x00185100, # PatientPosition
    0x00200013, # InstanceNumber
    0x00200032, # ImagePositionPatient
    0x00200037, # ImageOrientation
  ]
  for required_tag in required_tags:
    if required_tag not in dataset: # Need Instance for offset calculation
      raise InvalidDataset

  if dataset[0x00200037].VM != 6: # ImageOrientation
    raise InvalidDataset

  image_orientation: Tuple[float, float, float, float, float, float] = tuple(pos for pos in dataset.ImageOrientationPatient)

  if dataset[0x00200032].VM != 3:
    raise InvalidDataset

  head_first = dataset.PatientPosition.startswith('HF') # Head First
  if head_first:
    orientation = -1
  else:
    orientation = 1

  initial_position: Tuple[float, float, float] = tuple(pos for pos in dataset.ImagePositionPatient)

  return extrapolate_image_position_patient(
    dataset.SliceThickness,
    orientation,
    initial_position,
    image_orientation,
    dataset.InstanceNumber,
    slices
  )

def format_from_patient_name_str(patient_name: str) -> str:
  """Formats the dicom encoded patient name into a human displayable name

  Args:
      patient_name (str): A string with one or more ^ denominating different
      parts of the name

  Returns:
      A string without any ^ in, so not for storing in Dicom objects
  """

  split_name = patient_name.split('^')

  if(len(split_name) == 1):
    # Probbally a test name
    return patient_name.capitalize().strip()
  if(len(split_name) == 2):
    family_name, given_name,= split_name
    return f"{given_name.capitalize()} {family_name.capitalize()}".strip()
  if(len(split_name) == 3):
    family_name, given_name, middle_name = split_name
    return f"{given_name.capitalize()} {middle_name.capitalize()} {family_name.capitalize()}".strip()
  if(len(split_name) == 4):
    family_name, given_name, middle_name, prefix = split_name
    return f"{prefix} {given_name.capitalize()} {middle_name.capitalize()} {family_name.capitalize()}".strip()
  if(len(split_name) == 5):
    family_name, given_name, middle_name, prefix, suffix = split_name
    return f"{prefix} {given_name.capitalize()} {middle_name.capitalize()} {family_name.capitalize()} {suffix}".strip()
  raise ValueError("A Patient name can only contain 5 ^'s")

def format_from_patient_name(person_name: PersonName) -> str:
  return_str = ""
  if person_name.name_prefix:
    return_str += f"{person_name.name_prefix} "
  if person_name.given_name:
    return_str += f"{person_name.given_name.capitalize()} "
  if person_name.middle_name:
    return_str += f"{person_name.middle_name.capitalize()} "
  if person_name.family_name:
    return_str += f"{person_name.family_name.capitalize()} "
  if person_name.name_suffix:
    return_str += f"{person_name.name_suffix}"

  return return_str.strip()

def has_tags(dataset: Dataset, tags: Union[
                                        List[str],
                                        List[int],
                                        List[Tuple[int,int]],
                                        List[Union[int,str, Tuple[int,int]]],
                                      ]):
  def and_(a, b):
    return a and b

  def in_(tag):
    return tag in dataset
  return reduce(and_, map(in_, tags), True)

def sort_datasets(dataset: Dataset):
  """Sorting function for a collection of datasets. The order is determined by
  the instance number

  Args:
      dataset (Dataset): _description_

  Returns:
      int: _description_
  """
  if 'ImagePositionPatient' in dataset:


    return dataset.ImagePositionPatient[2]

  return dataset.InstanceNumber

from . import anonymization
from . import blueprints
from . import dicom_factory
from . import dimse
from . import lazy_dataset
from . import nifti
from . import sop_mapping

__all__ = [
  'gen_uid',
  'make_meta',
  'get_tag',
  'get_dicomnode_creator_header',
  'refresh_dataset_dict',
  'extrapolate_image_position_patient_dataset',
  'format_from_patient_name_str',
  'format_from_patient_name',
  'anonymization',
  'blueprints',
  'dicom_factory',
  'dimse',
  'lazy_dataset',
  'nifti',
  'sort_datasets',
  'sop_mapping',
]
