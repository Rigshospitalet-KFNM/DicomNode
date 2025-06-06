"""Library methods for manipulation of pydicom.dataset objects
"""

# Python Standard Library
from functools import reduce
from enum import Enum
from types import EllipsisType
from typing import Any,Callable, List, Literal, Iterable, Optional, Tuple, Union
# Third Party Libraries
import numpy
import nibabel
from pydicom import DataElement
from pydicom.dataset import Dataset, validate_file_meta
from pydicom.tag import BaseTag
from pydicom.uid import UID, generate_uid, ImplicitVRLittleEndian, ExplicitVRBigEndian, ExplicitVRLittleEndian
from pydicom.valuerep import PersonName

# Dicomnode packages
from dicomnode.constants import DICOMNODE_IMPLEMENTATION_UID, DICOMNODE_IMPLEMENTATION_NAME, DICOMNODE_VERSION, DICOMNODE_PRIVATE_TAG_HEADER
from dicomnode.lib.exceptions import InvalidDataset, MissingDatasets

PRIVATIZATION_VERSION = 1

# Ensure Correct loading
#import dicomnode.math as __math


class Reserved_Tags(Enum):
  PRIVATE_TAG_NAMES = 0xFE
  PRIVATE_TAG_VRS = 0xFF

def gen_uid() -> UID:
  """Generates a Unique identifier with Dicomnode's UID prefix

  Returns:
      UID: A Unique identifier with Dicomnode's prefix.
  """
  return generate_uid(prefix=DICOMNODE_IMPLEMENTATION_UID + '.')

def get_private_group_id(tag: BaseTag | int):
  return (tag & 0xFFFF0000) + ((tag >> 8) & 0xFF)

def get_private_group_tag(tag: BaseTag | int):
  return tag & 0xFFFFFF00

def is_private_group_tag(tag: BaseTag | int):
  return 0x0001_0000 & tag and not tag & 0xFF00


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

  position = [numpy.array(initial_position) + (slice_num - (image_number - 1)) * cross_vector for slice_num in numpy.arange(0, slices, 1, dtype=numpy.float64)]

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

def create_dicom_coordinate_system(spacial_reference, rm_shape=None, cm_shape=None):
  """Creates the values describing the spacial_reference's coordinate system for
  a dicom series

  Args:
      spacial_reference: This the object that contains the coordinate system,
        such as an affine matrix, a nifti object, an dicomnode image or a dicomnode space
      rm_shape (Tuple[int,int,int], optional): Row major shape container (Default for this
        library) Shape container with the format (Z,Y,X). Required if the
        spacial_reference is an affine and cm_shape is None. Defaults to None.
      cm_shape (Tuple[int,int,int], optional): Column major shape container (Default for Nifti)
        Shape container with the format (X,Y,Z). Required if the
        spacial_reference is an affine and rm_shape is None. Defaults to None.

  Raises:
      TypeError: If you do something boneheaded. Read the error and fix it...

  Returns:
      Tuple: [List[List[float]], tuple[floating[Any], floating[Any], floating[Any]], tuple[Any, ...], Tuple[float,float, float] -
        This tuple contains:
          * Points - This is a list of points where each member should be placed in 0x0020_0032
          * Voxel_dim - This is a tuple with the size of a pixel in (X,Y,Z) sizes
          * Orientation - The x and y vector that should be placed in x0020_0037
          * Starting point - The starting point of the coordinate system.
  """
  # These imports are just to prevent Circular imports
  from dicomnode.math.image import Image
  from dicomnode.math.space import Space

  if isinstance(spacial_reference, Space):
    if rm_shape is not None or cm_shape is not None:
      raise TypeError("When passing a space, you shouldn't pass a shape to the function")
    starting_point = spacial_reference.starting_point
    vec_x = spacial_reference.basis[0]
    vec_y = spacial_reference.basis[1]
    vec_z = spacial_reference.basis[2]
    extent = spacial_reference.extent

  elif isinstance(spacial_reference, Image):
    if rm_shape is not None or cm_shape is not None:
      raise TypeError("When passing an image, you shouldn't pass a shape to the function")

    starting_point = spacial_reference.space.starting_point
    vec_x = spacial_reference.space.basis[0]
    vec_y = spacial_reference.space.basis[1]
    vec_z = spacial_reference.space.basis[2]
    extent = spacial_reference.space.extent
  elif isinstance(spacial_reference, nibabel.nifti1.Nifti1Image) or isinstance(spacial_reference, nibabel.nifti2.Nifti2Image):
    if rm_shape is not None or cm_shape is not None:
      raise TypeError("When passing an Nifti object, you shouldn't pass a shape to the function")
    if spacial_reference.affine is None:
      raise TypeError("Unable to create coordinate system as input nifti doesn't contain any Affine")

    vec_x = spacial_reference.affine[0,:3]
    vec_y = spacial_reference.affine[1,:3]
    vec_z = spacial_reference.affine[2,:3]
    starting_point = spacial_reference.affine[:3, 3]
    extent = tuple(reversed(spacial_reference.shape))

  elif isinstance(spacial_reference, numpy.ndarray) and (rm_shape is not None and cm_shape is not None):
    if rm_shape is not None and cm_shape is not None:
      raise TypeError("You shouldn't pass both a shape and ww_shape")

    if spacial_reference.shape != (4,4):
      raise TypeError("The spacial reference is not an affine matrix")

    vec_x = spacial_reference[0,:3]
    vec_y = spacial_reference[1,:3]
    vec_z = spacial_reference[2,:3]
    starting_point = spacial_reference[:3, 3]
    if cm_shape is None:
      extent = tuple(reversed(spacial_reference.shape))
    else:
      extent = rm_shape
  else:
    raise TypeError(f"Unable able create dicom positions from {spacial_reference} of type: {type(spacial_reference)}")

  # Input parsing done and starting_point, vec_x, vec_y, vec_z and extent is defied

  pixel_size_x = numpy.linalg.norm(vec_x)
  pixel_size_y = numpy.linalg.norm(vec_y)
  pixel_size_z = numpy.linalg.norm(vec_z)

  orientation = numpy.concat((
    vec_x / pixel_size_x,
    vec_y / pixel_size_y
  ))

  starting_point = starting_point

  points = extrapolate_image_position_patient(
    float(pixel_size_z), numpy.sign(vec_z[2]),
        starting_point,
        orientation, 1, extent[2]
    )

  return points, (pixel_size_x, pixel_size_y, pixel_size_z), list(orientation), starting_point


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

def assess_single_series(datasets: Iterable[Dataset]) -> Optional[UID]:
  """Assess a collection of datasets to ensure, that they are of a single
  series. Throws if multiple series are inside of the collection.

  Args:
    datasets: Iterable[Dataset] - The collection of datasets to

  Returns:
    UID: Optional[UID] - None if no instance UID series in datasets, otherwise
                         The series uid of all the datasets

  Raises:
    InvalidDataset: If the collection contains multiple series.
    MissingDatasets: if the collection is empty
  """

  sentinel: None | EllipsisType | UID = ...

  for dataset in datasets:
    if sentinel is ...:
      sentinel = dataset.get('SeriesInstanceUID', None)
    else:
      if sentinel != dataset.get('SeriesInstanceUID', None):
        raise InvalidDataset("Collection contains multiple Series")

  if sentinel is ...:
    raise MissingDatasets("Dataset Container is empty")

  return sentinel


class ComparingDatasets:
  def __init__(self, dataset_1, dataset_2) -> None:
    self.dataset_1 = iter(dataset_1)
    self.dataset_2 = iter(dataset_2)

    self._next_tag_1: Optional[DataElement] = None
    self._next_tag_2: Optional[DataElement] = None

  def __iter__(self):
    return self

  def __next__(self):
    if self._next_tag_1 is None:
      try:
        self._next_tag_1 = next(self.dataset_1)
      except StopIteration:
        self._next_tag_1 = None

    if self._next_tag_2 is None:
      try:
        self._next_tag_2 = next(self.dataset_2)
      except StopIteration:
        self._next_tag_2 = None

    tag_1 = self._next_tag_1
    tag_2 = self._next_tag_2

    if(tag_1 is None and tag_2 is None):
      raise StopIteration #
    elif tag_1 is None:
      self._next_tag_2 = None
      return (None, tag_2)
    elif tag_2 is None:
      self._next_tag_1 = None
      return (tag_1, None)
    else:
      if tag_1.tag == tag_2.tag:
        self._next_tag_1 = None
        self._next_tag_2 = None
        return (tag_1, tag_2)
      elif tag_1.tag < tag_2.tag:
        self._next_tag_1 = None
        return (tag_1, None)
      else:
        self._next_tag_2 = None
        return (None, tag_2)

def add_private_tag(dataset: Dataset, data_element: DataElement):
  if not data_element.is_private:
    raise ValueError("The data_element is not private tag")

  if is_private_group_tag(data_element.tag):
    raise ValueError("Group Private tags are automatically, you don't have to add them")

  tag_group_id = get_private_group_id(data_element.tag)
  group_tag = get_private_group_tag(data_element.tag)
  group_tag_name = group_tag + Reserved_Tags.PRIVATE_TAG_NAMES.value
  group_tag_VR = group_tag + Reserved_Tags.PRIVATE_TAG_VRS.value

  if data_element.tag == group_tag_name:
    raise ValueError("Reserved tag collision, This tag is reserved for Private tag names by dicomnode")

  if data_element.tag == group_tag_VR:
    raise ValueError("Reserved tag collision, This tag is reserved for Private tag VR by dicomnode")

  if tag_group_id in dataset:
    group_owner: str = dataset[tag_group_id].value
    if not group_owner.startswith("Dicomnode"):
      raise ValueError("You're trying to add a private to a group that Dicomnode cannot claim ownership over.")
    if group_owner != DICOMNODE_PRIVATE_TAG_HEADER:
      raise ValueError("The tag group is owned by a different private tag version than the current one installed in the library")
  else:
    dataset.add_new(tag_group_id, 'LO', DICOMNODE_PRIVATE_TAG_HEADER)

  if group_tag_name in dataset:
    if dataset[group_tag_name].VM == 1:
      dataset[group_tag_name].value = [dataset[group_tag_name].value, data_element.name]
    else:
      dataset[group_tag_name].value.append(data_element.name)
  else:
    dataset.add_new(group_tag_name, 'LO', data_element.name)

  if group_tag_VR in dataset:
    if dataset[group_tag_VR].VM == 1:
      dataset[group_tag_VR].value = [dataset[group_tag_VR].value, data_element.VR]
    else:
      dataset[group_tag_VR].value.append(data_element.VR)
  else:
    dataset.add_new(group_tag_VR, 'LO', data_element.VR)

  dataset.add(data_element)


def sanity_check_dataset(dataset: Dataset) -> bool:
  """_summary_

  Args:
      dataset (Dataset): _description_

  Returns:
      bool: _description_
  """
  required_tags = [
    0x0008_0016,
    0x0008_0018,
    0x0020_000D,
    0x0020_000E
  ]

  def tag_in_dataset(tag):
    return tag in dataset

  tags_in_dataset = map(tag_in_dataset, required_tags)

  def and_(a, b):
    return a and b

  return reduce(and_, tags_in_dataset, True)

def print_difference_between_datasets(dataset_1 : Dataset, dataset_2: Dataset):
  for (tag_1, tag_2) in ComparingDatasets(dataset_1, dataset_2):
    if tag_1 is None and tag_2 is not None:
      print(f"Missing {tag_2.tag} from the first dataset")
    elif tag_2 is None and tag_1 is not None:
      print(f"Missing {tag_1.tag} from the second dataset")

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
